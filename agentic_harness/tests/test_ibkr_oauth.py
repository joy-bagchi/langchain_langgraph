from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
from starlette.testclient import TestClient
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_registration_request,
)
from mcp.shared.auth import OAuthClientMetadata, OAuthMetadata

from agentic_harness.ibkr_oauth import (
    CALLBACK_PATH,
    IBKROAuthCallbackError,
    IBKR_MCP_SERVER_URL,
    HarnessIBKROAuthConfig,
    create_ibkr_oauth_app,
    create_hosted_authorization_link,
    discover_ibkr_oauth_metadata,
    oauth_diagnostic_hooks,
    safe_endpoint_identity,
    sanitize_oauth_error_description,
    validate_callback,
)
from agentic_harness.ibkr_data_reader import READ_ONLY_SCOPE


def _config() -> HarnessIBKROAuthConfig:
    service_url = "https://harness.example.run.app"
    return HarnessIBKROAuthConfig(
        project_id="marketphysics",
        client_secret_id="harness-client",
        token_secret_id="harness-token",
        transaction_secret_id="harness-transaction",
        service_url=service_url,
        redirect_uri=service_url + CALLBACK_PATH,
        allowed_email="owner@example.com",
        identity_token_audience=service_url,
    )


def test_config_requires_service_owned_https_callback_and_separate_secrets() -> None:
    values = {
        "GOOGLE_CLOUD_PROJECT": "marketphysics",
        "HARNESS_IBKR_CLIENT_SECRET_ID": "harness-client",
        "HARNESS_IBKR_TOKEN_SECRET_ID": "harness-token",
        "HARNESS_IBKR_TRANSACTION_SECRET_ID": "harness-transaction",
        "HARNESS_IBKR_OAUTH_SERVICE_URL": "https://harness.example.run.app",
        "HARNESS_IBKR_OAUTH_REDIRECT_URI": "https://harness.example.run.app/ibkr/callback",
        "HARNESS_IBKR_OAUTH_ALLOWED_EMAIL": "OWNER@example.com",
    }
    assert HarnessIBKROAuthConfig.from_env(values) == _config()
    values["HARNESS_IBKR_OAUTH_REDIRECT_URI"] += "?secret=value"
    try:
        HarnessIBKROAuthConfig.from_env(values)
    except ValueError:
        pass
    else:
        raise AssertionError("callback query string was accepted")


def test_callback_is_state_bound_and_single_use() -> None:
    assert validate_callback(expected_state="expected", returned_state="expected", code="private-code", used=False) == "private-code"
    for kwargs in (
        {"expected_state": "expected", "returned_state": "other", "code": "private-code", "used": False},
        {"expected_state": "expected", "returned_state": "expected", "code": "private-code", "used": True},
        {"expected_state": "expected", "returned_state": "expected", "code": None, "used": False},
    ):
        try:
            validate_callback(**kwargs)
        except IBKROAuthCallbackError as exc:
            assert "private-code" not in str(exc)
        else:
            raise AssertionError("invalid callback was accepted")


def test_discovery_follows_observed_challenge_and_selects_advertised_dcr() -> None:
    async def exercise() -> None:
        context = SimpleNamespace(
            client_metadata=OAuthClientMetadata(
                redirect_uris=[_config().redirect_uri], application_type="web",
                token_endpoint_auth_method="none", scope="mcp.read", client_name="Harness read only",
            ),
        )
        auth = SimpleNamespace(context=context)
        requests: list[httpx.Request] = []
        protected_url = "https://api.ibkr.com/v1/api/mcp-public/.well-known/oauth-protected-resource"
        protected_urls = build_protected_resource_metadata_discovery_urls(protected_url, IBKR_MCP_SERVER_URL)
        oauth_urls = build_oauth_authorization_server_metadata_discovery_urls("https://api.ibkr.com", IBKR_MCP_SERVER_URL)

        async def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/v1/api/mcp-public":
                return httpx.Response(401, request=request, headers={
                    "WWW-Authenticate": f'Bearer resource_metadata="{protected_url}"'
                })
            if str(request.url) in protected_urls:
                return httpx.Response(200, request=request, json={
                    "resource": IBKR_MCP_SERVER_URL,
                    "authorization_servers": ["https://api.ibkr.com"],
                    "scopes_supported": ["mcp.read", "mcp.write"],
                })
            if str(request.url) in oauth_urls:
                return httpx.Response(200, request=request, json={
                    "issuer": "https://api.ibkr.com",
                    "authorization_endpoint": "https://api.ibkr.com/oauth2/authorize",
                    "token_endpoint": "https://api.ibkr.com/oauth2/api/v1/token",
                    "registration_endpoint": "https://api.ibkr.com/oauth2/register",
                    "token_endpoint_auth_methods_supported": ["none", "client_secret_basic"],
                    "scopes_supported": ["mcp.read", "mcp.write", "mcp.orders.submit"],
                })
            raise AssertionError(f"unexpected discovery path {request.url.path}")

        endpoint = await discover_ibkr_oauth_metadata(auth, transport=httpx.MockTransport(respond))
        assert endpoint == "https://api.ibkr.com/oauth2/register"
        assert str(auth.context.oauth_metadata.registration_endpoint) == endpoint
        assert auth.context.client_metadata.scope == READ_ONLY_SCOPE
        assert [request.url.path for request in requests] == [
            "/v1/api/mcp-public",
            "/v1/api/mcp-public/.well-known/oauth-protected-resource",
            "/.well-known/oauth-authorization-server",
        ]

    asyncio.run(exercise())


def test_captured_dcr_request_uses_advertised_url_web_app_and_only_mcp_read() -> None:
    async def exercise() -> None:
        endpoint = "https://api.ibkr.com/oauth2/register"
        metadata = OAuthClientMetadata(
            redirect_uris=[_config().redirect_uri], application_type="web",
            token_endpoint_auth_method="none", scope=READ_ONLY_SCOPE,
            client_name="Agentic Harness read-only MCP",
        )
        auth = SimpleNamespace(context=SimpleNamespace(
            client_metadata=metadata,
            oauth_metadata=OAuthMetadata.model_validate({
                "issuer": "https://api.ibkr.com",
                "authorization_endpoint": "https://api.ibkr.com/oauth2/authorize",
                "token_endpoint": "https://api.ibkr.com/oauth2/api/v1/token",
                "registration_endpoint": endpoint,
            }),
        ))
        captured: list[tuple[str, dict]] = []
        diagnostics: dict = {}

        async def capture(request: httpx.Request) -> httpx.Response:
            captured.append((str(request.url), json.loads(request.content)))
            return httpx.Response(201, request=request, json={"client_id": "offline-client"})

        fallback = create_client_registration_request(None, metadata, "https://api.ibkr.com")
        assert str(fallback.url) == "https://api.ibkr.com/register"
        request = create_client_registration_request(auth.context.oauth_metadata, metadata, "https://api.ibkr.com")
        assert str(request.url) == endpoint
        body = json.loads(request.content)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(capture),
            event_hooks=oauth_diagnostic_hooks(auth, registration_endpoint=endpoint, diagnostics=diagnostics),
        ) as client:
            response = await client.send(request)
        assert response.status_code == 201
        assert captured == [(endpoint, body)]
        assert body["application_type"] == "web"
        assert body["scope"] == "mcp.read"
        assert body["grant_types"] == ["authorization_code", "refresh_token"]
        assert body["response_types"] == ["code"]
        assert diagnostics["registration_endpoint"] == endpoint
        assert diagnostics["requested_scope"] == READ_ONLY_SCOPE
        assert diagnostics["grant_types"] == "authorization_code,refresh_token"
        assert diagnostics["response_types"] == "code"
        assert diagnostics["callback_uri"] == safe_endpoint_identity(_config().redirect_uri)

    asyncio.run(exercise())


def test_challenged_scope_cannot_expand_past_mcp_read() -> None:
    async def exercise() -> None:
        protected_url = "https://api.ibkr.com/v1/api/mcp-public/.well-known/oauth-protected-resource"

        async def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/api/mcp-public":
                return httpx.Response(401, request=request, headers={
                    "WWW-Authenticate": f'Bearer resource_metadata="{protected_url}", scope="mcp.read mcp.write"'
                })
            raise AssertionError("broader challenge must fail before metadata lookup")

        auth = SimpleNamespace(context=SimpleNamespace(client_metadata=OAuthClientMetadata(
            redirect_uris=[_config().redirect_uri], application_type="web", scope=READ_ONLY_SCOPE,
        )))
        from agentic_harness.ibkr_data_reader import IBKRScopeRejected

        try:
            await discover_ibkr_oauth_metadata(auth, transport=httpx.MockTransport(respond))
        except IBKRScopeRejected:
            return
        raise AssertionError("broader challenge scope was accepted")

    asyncio.run(exercise())


def test_dcr_rejects_broader_scope_and_safe_diagnostics_redact_secrets() -> None:
    assert safe_endpoint_identity("https://api.ibkr.com/oauth2/register?code=private") == "https://api.ibkr.com/oauth2/register"
    description = sanitize_oauth_error_description("bad https://provider.test/path?code=private-code access_token=token-value")
    assert "provider.test" not in description
    assert "private-code" not in description
    assert "token-value" not in description

    async def exercise() -> None:
        from agentic_harness.ibkr_data_reader import IBKRScopeRejected

        endpoint = "https://api.ibkr.com/oauth2/register"
        metadata = OAuthClientMetadata(
            redirect_uris=[_config().redirect_uri], application_type="web",
            token_endpoint_auth_method="none", scope="mcp.read mcp.write", client_name="Harness read only",
        )
        auth = SimpleNamespace(context=SimpleNamespace(client_metadata=metadata, oauth_metadata=None))
        hooks = oauth_diagnostic_hooks(auth, registration_endpoint=endpoint, diagnostics={})
        request = httpx.Request("POST", endpoint, json=metadata.model_dump(by_alias=True, mode="json", exclude_none=True))
        try:
            await hooks["request"][0](request)
        except IBKRScopeRejected:
            pass
        else:
            raise AssertionError("write scope passed the registration guard")

    asyncio.run(exercise())


def test_dcr_failure_keeps_only_safe_structured_diagnostics(caplog) -> None:
    async def exercise() -> dict:
        endpoint = "https://api.ibkr.com/oauth2/register"
        metadata = OAuthClientMetadata(
            redirect_uris=[_config().redirect_uri], application_type="web",
            token_endpoint_auth_method="none", scope=READ_ONLY_SCOPE,
            client_name="Agentic Harness read-only MCP",
        )
        auth = SimpleNamespace(context=SimpleNamespace(client_metadata=metadata, oauth_metadata=None))
        diagnostics: dict = {}
        body = metadata.model_dump(by_alias=True, mode="json", exclude_none=True)
        body["application_type"] = "web"

        async def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, request=request, json={
                "error": "invalid_client",
                "error_description": "denied at https://provider.invalid/?code=authorization-code-secret access_token=access-token-secret",
                "private_detail": "must not appear",
            })

        request = httpx.Request("POST", endpoint, json=body)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(reject),
            event_hooks=oauth_diagnostic_hooks(auth, registration_endpoint=endpoint, diagnostics=diagnostics),
        ) as client:
            await client.send(request)
        return diagnostics

    caplog.set_level(logging.WARNING)
    diagnostics = asyncio.run(exercise())
    assert diagnostics["stage"] == "dynamic_client_registration"
    assert diagnostics["registration_endpoint"] == "https://api.ibkr.com/oauth2/register"
    assert diagnostics["http_status"] == 403
    assert diagnostics["oauth_error_code"] == "invalid_client"
    assert diagnostics["requested_scope"] == READ_ONLY_SCOPE
    assert diagnostics["application_type"] == "web"
    assert diagnostics["grant_types"] == "authorization_code,refresh_token"
    assert diagnostics["response_types"] == "code"
    rendered = caplog.text + repr(diagnostics)
    for secret in ("authorization-code-secret", "access-token-secret", "private_detail", "provider.invalid"):
        assert secret not in rendered


def test_harness_oauth_app_is_disabled_by_default() -> None:
    app = create_ibkr_oauth_app(enabled=False)
    with TestClient(app) as client:
        assert client.get("/health").json() == {
            "status": "ok", "ibkr_tools_enabled": False, "oauth_enabled": False,
        }
        response = client.post("/ibkr/reconnect", json={"run_id": "run-1"})
    assert response.status_code == 503
    assert "authorization_url" not in response.json()


def test_authorization_route_checks_identity_before_secret_manager_or_ibkr() -> None:
    class Store:
        accessed = False

        def load_client_info(self):
            self.accessed = True

        def load_tokens(self):
            self.accessed = True

    store = Store()

    def reject_identity(_request, _config):
        raise PermissionError("private identity failure")

    app = create_ibkr_oauth_app(
        config=_config(), credential_store=store,
        identity_verifier=reject_identity, enabled=True,
    )
    with TestClient(app) as client:
        response = client.post("/ibkr/reconnect", json={"run_id": "run-1"})
    assert response.status_code == 401
    assert response.json() == {"error": "Harness identity is not authorized"}
    assert store.accessed is False


class _DurableOAuthStore:
    def __init__(self) -> None:
        self.client_info = {"client_id": "offline-client", "scope": READ_ONLY_SCOPE}
        self.tokens = {"access_token": "old-access-token", "refresh_token": "old-refresh-token", "scope": READ_ONLY_SCOPE}
        self.transaction = None

    def load_client_info(self):
        return self.client_info

    def load_tokens(self):
        return self.tokens

    def load_oauth_transaction(self):
        return dict(self.transaction) if self.transaction else None

    def persist_oauth_transaction(self, transaction):
        self.transaction = dict(transaction)

    def persist_tokens(self, tokens):
        self.tokens = dict(tokens)


class _WaitingWorkflow:
    def __init__(self):
        self.resumed = []

    def is_authorization_required(self, run_id):
        return run_id == "run-1"

    def resume(self, run_id):
        self.resumed.append(run_id)
        return {"status": "completed", "fresh_data": True}


def test_hosted_reconnect_persists_transaction_and_resumes_after_process_boundary() -> None:
    store = _DurableOAuthStore()
    lifecycle = _WaitingWorkflow()
    state = "offline-state-value"
    url = f"https://api.ibkr.com/oauth2/authorize?client_id=offline-client&state={state}&scope=mcp.read"

    async def prepare(run_id, _diagnostics):
        store.persist_oauth_transaction({
            "status": "prepared", "state": state, "code_verifier": "offline-pkce-verifier",
            "run_id": run_id, "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        })
        return url

    async def exchange(_store, _config, transaction, code):
        assert transaction["code_verifier"] == "offline-pkce-verifier"
        assert code == "one-time-code"
        tokens = {"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "scope": READ_ONLY_SCOPE}
        _store.persist_tokens(tokens)
        return tokens

    app = create_ibkr_oauth_app(
        config=_config(), credential_store=store, identity_verifier=lambda *_: None,
        workflow_lifecycle=lifecycle, authorization_preparer=prepare,
        code_exchanger=exchange, enabled=True,
    )
    with TestClient(app) as client:
        response = client.post("/ibkr/reconnect", json={"run_id": "run-1"})
    assert response.status_code == 200
    assert response.json()["authorization_url"] == url
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["requested_scope"] == READ_ONLY_SCOPE
    assert store.transaction["status"] == "prepared"

    # A new app instance models a Cloud Run revision/instance boundary. The
    # callback reconstructs the exchange from protected transaction storage.
    resumed_app = create_ibkr_oauth_app(
        config=_config(), credential_store=store, identity_verifier=lambda *_: None,
        workflow_lifecycle=lifecycle, code_exchanger=exchange, enabled=True,
    )
    with TestClient(resumed_app) as client:
        callback = client.get(f"{CALLBACK_PATH}?state={state}&code=one-time-code")
        replay = client.get(f"{CALLBACK_PATH}?state={state}&code=one-time-code")
    assert callback.status_code == 200
    assert "resumed with fresh data" in callback.text
    assert replay.status_code == 400
    assert lifecycle.resumed == ["run-1"]
    assert store.tokens["refresh_token"] == "rotated-refresh-token"
    assert store.transaction["status"] == "completed"
    assert "one-time-code" not in callback.text
    assert "rotated-access-token" not in callback.text


def test_hosted_authorization_url_uses_fresh_pkce_and_persists_verifier_before_delivery() -> None:
    store = _DurableOAuthStore()
    context = SimpleNamespace(
        client_metadata=SimpleNamespace(scope=READ_ONLY_SCOPE),
        client_info=SimpleNamespace(client_id="offline-client"),
        oauth_metadata=SimpleNamespace(authorization_endpoint="https://api.ibkr.com/oauth2/authorize"),
        protocol_version=None,
        should_include_resource_param=lambda _version: False,
        get_resource_url=lambda: IBKR_MCP_SERVER_URL,
    )
    url = asyncio.run(create_hosted_authorization_link(
        context, config=_config(), run_id="run-1", transaction_store=store,
    ))
    from urllib.parse import parse_qs, urlsplit

    query = parse_qs(urlsplit(url).query)
    assert query["scope"] == [READ_ONLY_SCOPE]
    assert query["redirect_uri"] == [_config().redirect_uri]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"] == [store.transaction["state"]]
    assert store.transaction["status"] == "prepared"
    assert store.transaction["run_id"] == "run-1"
    assert store.transaction["code_verifier"] not in url
    assert store.transaction["code_verifier"]


def test_hosted_callback_rejects_broader_grant_without_resuming() -> None:
    store = _DurableOAuthStore()
    store.persist_oauth_transaction({
        "status": "prepared", "state": "scope-state", "code_verifier": "verifier",
        "run_id": "run-1", "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    })
    lifecycle = _WaitingWorkflow()

    async def reject_broad_scope(*_args):
        return {"access_token": "must-not-escape", "scope": "mcp.read mcp.write"}

    app = create_ibkr_oauth_app(
        config=_config(), credential_store=store, identity_verifier=lambda *_: None,
        workflow_lifecycle=lifecycle, code_exchanger=reject_broad_scope, enabled=True,
    )
    with TestClient(app) as client:
        response = client.get(f"{CALLBACK_PATH}?state=scope-state&code=code-value")
    assert response.status_code == 403
    assert lifecycle.resumed == []
    assert "must-not-escape" not in response.text


def test_secret_manager_store_keeps_hosted_transaction_in_a_separate_secret() -> None:
    class FakeSecretManager:
        def __init__(self):
            self.payloads = {}

        def add_secret_version(self, request):
            self.payloads[request["parent"]] = request["payload"]["data"]

        def access_secret_version(self, request):
            from types import SimpleNamespace

            name = request["name"].removesuffix("/versions/latest")
            if name not in self.payloads:
                raise type("NotFound", (Exception,), {})()
            return SimpleNamespace(payload=SimpleNamespace(data=self.payloads[name]))

    from agentic_harness.ibkr_data_reader import SecretManagerIBKRCredentialStore

    client = FakeSecretManager()
    store = SecretManagerIBKRCredentialStore(
        project_id="p", client_secret_id="client", token_secret_id="tokens",
        transaction_secret_id="oauth-state", client=client,
    )
    transaction = {
        "status": "prepared", "state": "private-state", "code_verifier": "private-verifier",
        "run_id": "run-1", "created_at": "2026-10-02T00:00:00+00:00",
    }
    store.persist_oauth_transaction(transaction)
    assert store.load_oauth_transaction() == transaction
    store.persist_tokens({"access_token": "private-access", "refresh_token": "private-refresh", "scope": READ_ONLY_SCOPE})
    stored_tokens = store.load_tokens()
    assert stored_tokens["scope"] == READ_ONLY_SCOPE
    assert stored_tokens["harness_obtained_at"]
    assert set(client.payloads) == {
        "projects/p/secrets/oauth-state", "projects/p/secrets/tokens",
    }


def test_generic_remote_mcp_client_hides_catalog_and_requires_reviewed_bindings() -> None:
    from agentic_harness.remote_mcp import RemoteMCPClient, RemoteMCPProfile

    client = RemoteMCPClient(RemoteMCPProfile(IBKR_MCP_SERVER_URL))
    assert not hasattr(client, "list_tools")
    assert not hasattr(client, "list_resources")
    assert not hasattr(client, "call_tool")
    try:
        RemoteMCPProfile(IBKR_MCP_SERVER_URL, "mcp.read mcp.write")
    except Exception as exc:
        from agentic_harness.remote_mcp import RemoteMCPScopeError

        assert isinstance(exc, RemoteMCPScopeError)
    else:
        raise AssertionError("remote profile accepted a broader grant")

    from agentic_harness.agentic_os.tool_service import RegisteredToolService
    from agentic_harness.ibkr_oauth import HarnessIBKRMCPDataProvider

    unmapped = HarnessIBKRMCPDataProvider(_DurableOAuthStore(), redirect_uri=_config().redirect_uri)
    registered = RegisteredToolService.with_defaults(ibkr_data_reader_provider=unmapped).list_tools()
    assert not any(tool.metadata.get("tool_type") == "ibkr_data_reader" for tool in registered)


def test_sdk_refresh_rotates_tokens_and_persists_only_exact_read_scope() -> None:
    from agentic_harness.ibkr_oauth import HarnessIBKRMCPDataProvider

    class Store:
        def __init__(self):
            self.client_info = {
                "client_id": "offline-client", "redirect_uris": [_config().redirect_uri],
                "application_type": "web", "scope": READ_ONLY_SCOPE,
                "token_endpoint_auth_method": "none",
            }
            self.tokens = {
                "access_token": "expired-access-token", "refresh_token": "old-refresh-token",
                "scope": READ_ONLY_SCOPE, "expires_in": 0,
            }

        def load_client_info(self):
            return dict(self.client_info)

        def load_tokens(self):
            return dict(self.tokens)

        def persist_tokens(self, tokens):
            self.tokens = dict(tokens)
            self.tokens["harness_obtained_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

        def persist_client_info(self, client_info):
            self.client_info = dict(client_info)

    store = Store()
    provider = HarnessIBKRMCPDataProvider(store, redirect_uri=_config().redirect_uri)
    protected_url = "https://api.ibkr.com/v1/api/mcp-public/.well-known/oauth-protected-resource"
    protected_urls = set(build_protected_resource_metadata_discovery_urls(protected_url, IBKR_MCP_SERVER_URL))
    oauth_urls = set(build_oauth_authorization_server_metadata_discovery_urls("https://api.ibkr.com", IBKR_MCP_SERVER_URL))
    requests = []
    granted_scope = {"value": READ_ONLY_SCOPE}

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/api/mcp-public":
            return httpx.Response(401, request=request, headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{protected_url}", scope="mcp.read"'
            })
        if str(request.url) in protected_urls:
            return httpx.Response(200, request=request, json={
                "resource": IBKR_MCP_SERVER_URL,
                "authorization_servers": ["https://api.ibkr.com"],
                "scopes_supported": ["mcp.read", "mcp.write"],
            })
        if str(request.url) in oauth_urls:
            return httpx.Response(200, request=request, json={
                "issuer": "https://api.ibkr.com",
                "authorization_endpoint": "https://api.ibkr.com/oauth2/authorize",
                "token_endpoint": "https://api.ibkr.com/oauth2/api/v1/token",
                "registration_endpoint": "https://api.ibkr.com/oauth2/register",
            })
        if request.url.path == "/oauth2/api/v1/token":
            return httpx.Response(200, request=request, json={
                "access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token",
                "token_type": "Bearer", "expires_in": 3600, "scope": granted_scope["value"],
            })
        raise AssertionError(f"unexpected refresh request path {request.url.path}")

    from agentic_harness.ibkr_data_reader import IBKRScopeRejected

    transport = httpx.MockTransport(respond)
    granted_scope["value"] = "mcp.read mcp.write"
    try:
        asyncio.run(provider._refresh_tokens(transport=transport))
    except IBKRScopeRejected:
        pass
    else:
        raise AssertionError("refresh accepted a broader token response")
    assert store.tokens["refresh_token"] == "old-refresh-token"

    granted_scope["value"] = READ_ONLY_SCOPE
    result = asyncio.run(provider._refresh_tokens(transport=transport))
    assert result is not None
    assert result.tokens["access_token"] == "rotated-access-token"
    assert result.tokens["refresh_token"] == "rotated-refresh-token"
    assert store.tokens["refresh_token"] == "rotated-refresh-token"
    assert store.tokens["scope"] == READ_ONLY_SCOPE
    assert store.tokens["harness_obtained_at"]
    assert len([request for request in requests if request.method == "POST"]) == 2
