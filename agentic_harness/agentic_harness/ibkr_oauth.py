"""Harness-owned, read-only IBKR MCP OAuth lifecycle.

This module handles discovery, dynamic client registration, callback validation,
and protected credential storage. It deliberately does not inspect or invoke
the authenticated IBKR tool catalog. Provider action mapping stays disabled
until the authenticated schemas have been reviewed.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hmac
import json
import logging
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from agentic_harness.ibkr_data_reader import (
    IBKRAuthorizationRequired,
    IBKRCredentials,
    IBKRDataReaderError,
    IBKRScopeRejected,
    IBKRStorageUnavailable,
    READ_ONLY_SCOPE,
    SecretManagerIBKRCredentialStore,
    _require_exact_read_scope,
)
from agentic_harness.remote_mcp import (
    RemoteMCPClient,
    RemoteMCPProfile,
    RemoteMCPScopeError,
    require_exact_scope,
)


IBKR_MCP_SERVER_URL = "https://api.ibkr.com/v1/api/mcp-public"
CALLBACK_PATH = "/ibkr/callback"
_IBKR_HOST = "api.ibkr.com"
_SECRET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_LOGGER = logging.getLogger("agentic_harness.ibkr_oauth")
# The pinned MCP fork may include response text in exception messages.
logging.getLogger("mcp.client.auth.oauth2").disabled = True


class IBKROAuthConfigurationError(ValueError):
    """Trusted runtime OAuth configuration is missing or unsafe."""


class IBKROAuthCallbackError(ValueError):
    """OAuth callback state is invalid, replayed, or incomplete."""


class IBKRProviderMappingUnavailable(IBKRDataReaderError):
    """Authenticated IBKR tool schemas have not been reviewed and mapped."""


def _require_ibkr_read_scope(scope: str | None) -> None:
    try:
        require_exact_scope(scope, expected_scope=READ_ONLY_SCOPE)
    except RemoteMCPScopeError as exc:
        raise IBKRScopeRejected("IBKR grant must contain exactly mcp.read") from exc


def safe_endpoint_identity(value: object) -> str:
    """Return an HTTPS endpoint identity with query, fragment, and userinfo removed."""
    try:
        parts = urlsplit(str(value))
    except (TypeError, ValueError):
        return "unavailable"
    if parts.scheme != "https" or not parts.hostname:
        return "unavailable"
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parts.port
    if port and port != 443:
        host = f"{host}:{port}"
    path = re.sub(r"(?i)/(?:[^/]*(?:token|secret|code|state)[^/]*)", "/[REDACTED]", parts.path or "/")
    return f"https://{host}{path}"


def sanitize_oauth_error_description(value: object) -> str:
    """Keep concise diagnostic text while removing URLs and credential-like values."""
    if not isinstance(value, str) or not value.strip():
        return "unavailable"
    description = re.sub(r"https?://[^\s\"'<>]+", "[URL]", value)
    description = re.sub(
        r"(?i)\b(scope|authorization|access|refresh|id|client)?[_ -]?(code|token|secret|state|credential|key)\b\s*[:=]\s*[^\s,;]+",
        r"\1\2=[REDACTED]",
        description,
    )
    description = re.sub(r"(?i)([?&](?:code|token|state|secret|key|client_id)=)[^&\s]+", r"\1[REDACTED]", description)
    description = re.sub(r"\b[A-Za-z0-9_+/=-]{24,}\b", "[REDACTED]", description)
    description = re.sub(r"[\r\n\t]+", " ", description)
    description = re.sub(r"[^A-Za-z0-9 .,;:_()/\-\[\]=]", " ", description)
    return re.sub(r"\s+", " ", description).strip()[:240] or "unavailable"


def validate_callback(*, expected_state: str | None, returned_state: str | None, code: str | None, used: bool) -> str:
    """Validate one callback without returning state or code in an error."""
    if used:
        raise IBKROAuthCallbackError("OAuth callback has already been used")
    if not expected_state or not returned_state or not hmac.compare_digest(expected_state, returned_state):
        raise IBKROAuthCallbackError("OAuth callback state did not match the pending authorization")
    if not code:
        raise IBKROAuthCallbackError("OAuth callback did not contain an authorization code")
    return code


@dataclass(frozen=True, slots=True)
class HarnessIBKROAuthConfig:
    project_id: str
    client_secret_id: str
    token_secret_id: str
    transaction_secret_id: str
    service_url: str
    redirect_uri: str
    allowed_email: str
    identity_token_audience: str
    callback_timeout_seconds: int = 600

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "HarnessIBKROAuthConfig":
        values = os.environ if env is None else env
        names = {
            "project_id": "GOOGLE_CLOUD_PROJECT",
            "client_secret_id": "HARNESS_IBKR_CLIENT_SECRET_ID",
            "token_secret_id": "HARNESS_IBKR_TOKEN_SECRET_ID",
            "transaction_secret_id": "HARNESS_IBKR_TRANSACTION_SECRET_ID",
            "service_url": "HARNESS_IBKR_OAUTH_SERVICE_URL",
            "redirect_uri": "HARNESS_IBKR_OAUTH_REDIRECT_URI",
            "allowed_email": "HARNESS_IBKR_OAUTH_ALLOWED_EMAIL",
        }
        resolved = {field: values.get(name, "").strip() for field, name in names.items()}
        missing = [names[field] for field, value in resolved.items() if not value]
        if missing:
            raise IBKROAuthConfigurationError("Missing Harness IBKR configuration: " + ", ".join(missing))
        service_url = resolved["service_url"].rstrip("/")
        callback_uri = resolved["redirect_uri"]
        audience = values.get("HARNESS_IBKR_OAUTH_IDENTITY_AUDIENCE", service_url).strip()
        service = urlsplit(service_url)
        callback = urlsplit(callback_uri)
        if service.scheme != "https" or not service.netloc or service.path or service.query or service.fragment:
            raise IBKROAuthConfigurationError("Harness OAuth service URL must be an HTTPS origin")
        if callback_uri != service_url + CALLBACK_PATH or callback.query or callback.fragment or callback.username or callback.password:
            raise IBKROAuthConfigurationError("Harness OAuth callback must exactly match the configured service callback")
        if audience != service_url:
            raise IBKROAuthConfigurationError("Harness OAuth identity audience must equal its service URL")
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", resolved["allowed_email"].lower()):
            raise IBKROAuthConfigurationError("Harness OAuth allowlist must contain one email address")
        secret_ids = [resolved[field] for field in ("client_secret_id", "token_secret_id", "transaction_secret_id")]
        if len(set(secret_ids)) != len(secret_ids):
            raise IBKROAuthConfigurationError("Harness OAuth client, token, and transaction data require separate secrets")
        if any(not _SECRET_ID_RE.fullmatch(resolved[field]) for field in ("client_secret_id", "token_secret_id", "transaction_secret_id")):
            raise IBKROAuthConfigurationError("Harness Secret Manager IDs contain unsupported characters")
        try:
            timeout = int(values.get("HARNESS_IBKR_OAUTH_CALLBACK_TIMEOUT_SECONDS", "600"))
        except ValueError as exc:
            raise IBKROAuthConfigurationError("Harness OAuth callback timeout must be 60 to 900 seconds") from exc
        if not 60 <= timeout <= 900:
            raise IBKROAuthConfigurationError("Harness OAuth callback timeout must be 60 to 900 seconds")
        return cls(**{**resolved, "service_url": service_url, "identity_token_audience": audience,
                      "allowed_email": resolved["allowed_email"].lower(), "callback_timeout_seconds": timeout})


class HarnessMCPTokenStorage:
    """MCP SDK storage adapter over Harness's versioned Secret Manager store."""

    def __init__(self, store: SecretManagerIBKRCredentialStore) -> None:
        self.store = store

    async def get_tokens(self) -> Any | None:
        from mcp.shared.auth import OAuthToken

        value = await asyncio.to_thread(self.store.load_tokens)
        if value is None:
            return None
        _require_exact_read_scope(IBKRCredentials(client_info={}, tokens=value))
        return OAuthToken.model_validate(value)

    async def set_tokens(self, tokens: Any) -> None:
        value = tokens.model_dump(mode="json", exclude_none=True)
        _require_exact_read_scope(IBKRCredentials(client_info={}, tokens=value))
        await asyncio.to_thread(self.store.persist_tokens, value)

    async def get_client_info(self) -> Any | None:
        from mcp.shared.auth import OAuthClientInformationFull

        value = await asyncio.to_thread(self.store.load_client_info)
        if value is None:
            return None
        scope = value.get("scope")
        if scope and set(str(scope).split()) != {READ_ONLY_SCOPE}:
            raise IBKRScopeRejected("IBKR client registration must stay within mcp.read")
        return OAuthClientInformationFull.model_validate(value)

    async def set_client_info(self, client_info: Any) -> None:
        value = client_info.model_dump(mode="json", exclude_none=True)
        scope = value.get("scope")
        if scope and set(str(scope).split()) != {READ_ONLY_SCOPE}:
            raise IBKRScopeRejected("IBKR client registration must stay within mcp.read")
        await asyncio.to_thread(self.store.persist_client_info, value)


def _safe_ibkr_url(value: object, label: str) -> httpx.URL:
    url = httpx.URL(str(value))
    if url.scheme != "https" or url.host != _IBKR_HOST or url.port not in (None, 443) or url.username or url.password or url.query or url.fragment:
        raise IBKROAuthConfigurationError(f"IBKR {label} is outside the trusted origin")
    return url


def _safe_metadata_list(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "not_advertised"
    if any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9:._-]{1,100}", item) for item in value):
        return "invalid"
    return ",".join(value)


async def create_hosted_authorization_link(
    auth_context: Any,
    *,
    config: HarnessIBKROAuthConfig,
    run_id: str,
    transaction_store: SecretManagerIBKRCredentialStore,
) -> str:
    """Build a hosted PKCE link and persist its one-use state before delivery."""
    from mcp.client.auth.oauth2 import PKCEParameters

    _require_ibkr_read_scope(auth_context.client_metadata.scope)
    if not auth_context.client_info or not auth_context.oauth_metadata:
        raise IBKROAuthConfigurationError("IBKR OAuth metadata is incomplete")
    endpoint = _safe_ibkr_url(auth_context.oauth_metadata.authorization_endpoint, "authorization endpoint")
    pkce = PKCEParameters.generate()
    state = secrets.token_urlsafe(32)
    params = {
        "response_type": "code", "client_id": auth_context.client_info.client_id,
        "redirect_uri": config.redirect_uri, "state": state,
        "code_challenge": pkce.code_challenge, "code_challenge_method": "S256",
        "scope": READ_ONLY_SCOPE,
    }
    if auth_context.should_include_resource_param(auth_context.protocol_version):
        params["resource"] = auth_context.get_resource_url()
    transaction = {
        "status": "prepared", "state": state,
        "code_verifier": pkce.code_verifier, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await asyncio.to_thread(transaction_store.persist_oauth_transaction, transaction)
    return f"{endpoint}?{urlencode(params)}"


async def discover_ibkr_oauth_metadata(auth: Any, *, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Follow IBKR's 401 resource-metadata challenge and require advertised DCR."""
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
        extract_resource_metadata_from_www_auth,
        extract_scope_from_www_auth,
    )
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

    async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(20, read=30)) as client:
        challenge = await client.get(IBKR_MCP_SERVER_URL)
        resource_url = extract_resource_metadata_from_www_auth(challenge)
        challenge_scope = extract_scope_from_www_auth(challenge)
        if challenge.status_code != 401 or not resource_url:
            raise IBKROAuthConfigurationError("IBKR did not provide the expected protected-resource challenge")
        if challenge_scope:
            _require_ibkr_read_scope(challenge_scope)
        protected = None
        for candidate_url in build_protected_resource_metadata_discovery_urls(resource_url, IBKR_MCP_SERVER_URL):
            endpoint = _safe_ibkr_url(candidate_url, "protected-resource metadata endpoint")
            response = await client.get(endpoint)
            if response.status_code != 200:
                if response.status_code < 400 or response.status_code >= 500:
                    raise IBKROAuthConfigurationError("IBKR protected-resource metadata is unavailable")
                continue
            try:
                candidate = ProtectedResourceMetadata.model_validate_json(response.content)
            except Exception:
                continue
            if str(candidate.resource).rstrip("/") != IBKR_MCP_SERVER_URL.rstrip("/"):
                raise IBKROAuthConfigurationError("IBKR metadata named a different protected resource")
            protected = candidate
            break
        if protected is None:
            raise IBKROAuthConfigurationError("IBKR protected-resource metadata was not discoverable")
        if READ_ONLY_SCOPE not in (protected.scopes_supported or []):
            raise IBKROAuthConfigurationError("IBKR protected-resource metadata does not advertise mcp.read")
        auth_server = _safe_ibkr_url(protected.authorization_servers[0], "authorization server")
        oauth = None
        raw_oauth: dict[str, Any] | None = None
        for candidate_url in build_oauth_authorization_server_metadata_discovery_urls(str(auth_server), IBKR_MCP_SERVER_URL):
            endpoint = _safe_ibkr_url(candidate_url, "authorization-server metadata endpoint")
            response = await client.get(endpoint)
            if response.status_code != 200:
                if response.status_code < 400 or response.status_code >= 500:
                    raise IBKROAuthConfigurationError("IBKR authorization-server metadata is unavailable")
                continue
            try:
                raw = json.loads(response.content)
                candidate = OAuthMetadata.model_validate(raw)
            except Exception:
                continue
            if candidate.registration_endpoint is None:
                continue
            oauth, raw_oauth = candidate, raw
            break
        if oauth is None or raw_oauth is None:
            raise IBKROAuthConfigurationError("IBKR did not advertise a dynamic registration endpoint")
        issuer = _safe_ibkr_url(oauth.issuer, "issuer")
        registration = _safe_ibkr_url(oauth.registration_endpoint, "registration endpoint")
        if str(issuer).rstrip("/") != str(auth_server).rstrip("/"):
            raise IBKROAuthConfigurationError("IBKR authorization-server issuer did not match discovery")
        auth.context.protected_resource_metadata = protected
        auth.context.auth_server_url = str(auth_server)
        auth.context.oauth_metadata = oauth
        auth.context.client_metadata.scope = READ_ONLY_SCOPE
        _LOGGER.info(
            "IBKR OAuth discovery status=%s resource_metadata=%s registration_endpoint=%s requested_scope=%s advertised_scopes=%s",
            challenge.status_code,
            safe_endpoint_identity(resource_url),
            safe_endpoint_identity(registration),
            READ_ONLY_SCOPE,
            _safe_metadata_list(protected.scopes_supported),
        )
        return str(registration)


def oauth_diagnostic_hooks(auth: Any, *, registration_endpoint: str, diagnostics: dict[str, Any]) -> dict[str, list[Callable[..., Any]]]:
    """Enforce exact registration metadata and retain only safe failure fields."""
    advertised = _safe_ibkr_url(registration_endpoint, "registration endpoint")

    async def before_request(request: httpx.Request) -> None:
        if request.method.upper() != "POST":
            return
        try:
            body = json.loads(request.content)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return
        if not isinstance(body, dict) or "redirect_uris" not in body:
            return
        expected = auth.context.client_metadata.model_dump(by_alias=True, mode="json", exclude_none=True)
        # application_type was added explicitly in the pinned SDK fork. Keep
        # the Harness contract exact even when importing this module with an
        # older SDK during offline linting.
        expected["application_type"] = "web"
        if body != expected or body.get("scope") != READ_ONLY_SCOPE or body.get("application_type") != "web":
            raise IBKRScopeRejected("IBKR registration did not match the Harness web/mcp.read contract")
        if request.url != advertised:
            # The pinned fork supports advertised metadata. Correct only its
            # known SDK fallback; never select or construct another endpoint.
            request.url = advertised
        diagnostics["registration_endpoint"] = safe_endpoint_identity(advertised)
        diagnostics["requested_scope"] = READ_ONLY_SCOPE
        diagnostics["application_type"] = "web"
        diagnostics["grant_types"] = _safe_metadata_list(body.get("grant_types"))
        diagnostics["response_types"] = _safe_metadata_list(body.get("response_types"))
        diagnostics["callback_uri"] = safe_endpoint_identity(body["redirect_uris"][0]) if len(body["redirect_uris"]) == 1 else "invalid"
        request.extensions["harness_ibkr_dcr"] = True

    async def after_response(response: httpx.Response) -> None:
        if not response.request.extensions.pop("harness_ibkr_dcr", False) or response.status_code < 400:
            return
        diagnostics["http_status"] = response.status_code
        diagnostics["stage"] = "dynamic_client_registration"
        try:
            await response.aread()
            body = json.loads(response.content)
            if isinstance(body, dict):
                code = body.get("error")
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", code):
                    diagnostics["oauth_error_code"] = code
                diagnostics["oauth_error_description"] = sanitize_oauth_error_description(body.get("error_description"))
        except Exception:
            diagnostics["oauth_error_code"] = "unavailable"
            diagnostics["oauth_error_description"] = "unavailable"
        _LOGGER.warning(
            "IBKR DCR failed endpoint=%s http_status=%s oauth_error_code=%s oauth_error_description=%s requested_scope=%s application_type=%s grant_types=%s response_types=%s callback_uri=%s",
            diagnostics.get("registration_endpoint", "unavailable"), diagnostics.get("http_status", "unavailable"),
            diagnostics.get("oauth_error_code", "unavailable"), diagnostics.get("oauth_error_description", "unavailable"),
            diagnostics.get("requested_scope", READ_ONLY_SCOPE), diagnostics.get("application_type", "web"),
            diagnostics.get("grant_types", "unavailable"), diagnostics.get("response_types", "unavailable"),
            diagnostics.get("callback_uri", "unavailable"),
        )

    return {"request": [before_request], "response": [after_response]}


class _AuthorizationLinkReady(Exception):
    """Internal control signal after a durable authorization transaction is saved."""


class HarnessWorkflowLifecycle:
    """Resume workflow runs through Harness's existing persisted run store."""

    def __init__(self, *, storage_root: str | None = None, database_url: str | None = None) -> None:
        self.storage_root = storage_root or os.environ.get("AGENTIC_HARNESS_STORAGE_ROOT", ".workflow_memory")
        self.database_url = database_url or os.environ.get("AGENTIC_HARNESS_DB_URL")

    def is_authorization_required(self, run_id: str) -> bool:
        from agentic_harness.stores import WorkflowRunStore

        try:
            state = WorkflowRunStore(self.storage_root, database_url=self.database_url).load_state(run_id)
        except (FileNotFoundError, ValueError, OSError):
            return False
        return (
            state.get("status") == "authorization_required"
            and (state.get("pending_authorization") or {}).get("tool_type") == "ibkr_data_reader"
        )

    def resume(self, run_id: str) -> dict[str, Any]:
        from agentic_harness.runtime import resume_workflow

        return resume_workflow(
            run_id,
            storage_root=self.storage_root,
            database_url=self.database_url,
            langsmith_tracing=False,
        )


def _safe_failure_tree(exc: BaseException) -> tuple[list[str], list[int]]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    types: list[str] = []
    statuses: set[int] = set()
    while pending and len(types) < 16:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        cls = type(current)
        type_name = f"{cls.__module__}.{cls.__qualname__}"
        types.append(type_name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", type_name) else "unknown.Exception")
        response = getattr(current, "response", None)
        for status in (getattr(current, "status_code", None), getattr(response, "status_code", None)):
            if type(status) is int and 100 <= status <= 599:
                statuses.add(status)
        if type(current).__name__ == "OAuthRegistrationError":
            match = re.match(r"Registration failed:\s*([1-5][0-9]{2})(?:\b| )", str(current))
            if match:
                statuses.add(int(match.group(1)))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    if pending:
        types.append("truncated.ExceptionTree")
    return types, sorted(statuses)


def verify_google_identity(request: Any, config: HarnessIBKROAuthConfig) -> None:
    """Require an identity token for this service and the configured account."""
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise PermissionError("A valid Harness identity token is required")
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(token, GoogleAuthRequest(), config.identity_token_audience)
    except Exception as exc:
        raise PermissionError("A valid Harness identity token is required") from exc
    if claims.get("email_verified") is not True or claims.get("email", "").lower() != config.allowed_email:
        raise PermissionError("The Harness identity is not authorized for IBKR consent")


def create_ibkr_oauth_app(
    config: HarnessIBKROAuthConfig | None = None,
    *,
    credential_store: SecretManagerIBKRCredentialStore | None = None,
    identity_verifier: Callable[[Any, HarnessIBKROAuthConfig], None] | None = None,
    workflow_lifecycle: Any | None = None,
    authorization_preparer: Callable[[str, dict[str, Any]], Any] | None = None,
    code_exchanger: Callable[[SecretManagerIBKRCredentialStore, HarnessIBKROAuthConfig, dict[str, Any], str], Any] | None = None,
    enabled: bool | None = None,
) -> Any:
    """Create Harness's protected Reconnect and public OAuth callback routes."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    is_enabled = os.environ.get("HARNESS_IBKR_OAUTH_ENABLED", "false").lower() == "true" if enabled is None else enabled
    if is_enabled and config is None:
        config = HarnessIBKROAuthConfig.from_env()
    if is_enabled and config is None:  # pragma: no cover
        raise IBKROAuthConfigurationError("Harness IBKR OAuth configuration is missing")

    app = Starlette()
    app.state.enabled = is_enabled
    app.state.config = config
    app.state.store = credential_store
    app.state.identity_verifier = identity_verifier or verify_google_identity
    app.state.workflow_lifecycle = workflow_lifecycle or HarnessWorkflowLifecycle()
    app.state.authorization_preparer = authorization_preparer
    app.state.code_exchanger = code_exchanger
    app.state.prepared_authorization_url = None
    app.state.lock = asyncio.Lock()
    if is_enabled and app.state.store is None:
        assert config is not None
        app.state.store = SecretManagerIBKRCredentialStore(
            project_id=config.project_id,
            client_secret_id=config.client_secret_id,
            token_secret_id=config.token_secret_id,
            transaction_secret_id=config.transaction_secret_id,
        )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "ibkr_tools_enabled": False, "oauth_enabled": is_enabled})

    async def begin_reconnect(request: Request) -> JSONResponse:
        if not is_enabled or config is None:
            return JSONResponse({"error": "Harness IBKR OAuth is not enabled"}, status_code=503)
        try:
            app.state.identity_verifier(request, config)
        except PermissionError:
            return JSONResponse({"error": "Harness identity is not authorized"}, status_code=401)
        try:
            body = await request.json()
            run_id = body.get("run_id") if isinstance(body, dict) else None
        except Exception:
            run_id = None
        setup_only = run_id is None
        if setup_only:
            # Initial integration consent has no blocked workflow yet. Later
            # Reconnect actions must carry a durable authorization_required run.
            run_id = "ibkr-integration-setup"
        elif run_id == "ibkr-integration-setup":
            return JSONResponse({"error": "The integration setup run_id is reserved"}, status_code=400)
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
            return JSONResponse({"error": "A valid authorization-required run_id is required"}, status_code=400)
        if not setup_only and not app.state.workflow_lifecycle.is_authorization_required(run_id):
            return JSONResponse({"error": "This workflow is not waiting for IBKR authorization"}, status_code=409)
        store: SecretManagerIBKRCredentialStore = app.state.store
        try:
            await asyncio.to_thread(store.load_client_info)
            await asyncio.to_thread(store.load_tokens)
        except (IBKRScopeRejected, IBKRStorageUnavailable):
            return JSONResponse({"error": "Protected IBKR credential storage is unavailable or rejected"}, status_code=503)
        except Exception:
            return JSONResponse({"error": "Protected IBKR credential storage is unavailable"}, status_code=503)

        async with app.state.lock:
            diagnostics: dict[str, Any] = {}
            authorization_url: str | None = None
            try:
                preparer = app.state.authorization_preparer or prepare_authorization
                authorization_url = await preparer(run_id, diagnostics)
            except _AuthorizationLinkReady:
                # The SDK auth task is intentionally aborted after durable
                # transaction creation; the callback completes the exchange on
                # any Cloud Run instance using protected PKCE state.
                authorization_url = app.state.prepared_authorization_url
            except Exception as exc:
                if app.state.prepared_authorization_url:
                    # Some SDK transports wrap the hosted control signal in an
                    # ExceptionGroup. The durable record and URL are already
                    # saved, so treat only that known post-registration path as
                    # successful preparation.
                    authorization_url = app.state.prepared_authorization_url
                else:
                    types, statuses = _safe_failure_tree(exc)
                    diagnostics.setdefault("stage", "mcp_remote_initialize")
                    diagnostics.setdefault("exception_types", types)
                    if "http_status" not in diagnostics and statuses:
                        diagnostics["http_status"] = statuses[-1]
                    _LOGGER.warning(
                        "Harness remote MCP authorization preparation failed stage=%s nested_exception_types=%s http_status=%s",
                        diagnostics.get("stage", "mcp_remote_initialize"), ",".join(types),
                        ",".join(str(value) for value in statuses) or "none",
                    )
                    safe = {key: value for key, value in diagnostics.items() if key in {
                        "stage", "registration_endpoint", "http_status", "oauth_error_code",
                        "oauth_error_description", "requested_scope", "application_type",
                        "grant_types", "response_types", "callback_uri", "exception_types",
                    }}
                    return JSONResponse({"error": "Harness could not prepare IBKR consent", "diagnostics": safe}, status_code=502)
            finally:
                app.state.prepared_authorization_url = None
            if not authorization_url:
                return JSONResponse({"error": "Harness could not prepare IBKR consent"}, status_code=502)
            return JSONResponse({
                "authorization_url": authorization_url,
                "requested_scope": READ_ONLY_SCOPE,
                "callback_uri": config.redirect_uri,
                "expires_in_seconds": config.callback_timeout_seconds,
                "ibkr_tools_enabled": False,
            }, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    async def receive_callback(request: Request) -> PlainTextResponse:
        if not is_enabled or config is None:
            return PlainTextResponse("Harness IBKR callback is not enabled", status_code=503)
        returned_state = request.query_params.get("state")
        store: SecretManagerIBKRCredentialStore = app.state.store
        try:
            transaction = await asyncio.to_thread(store.load_oauth_transaction)
        except Exception:
            transaction = None
        if not transaction or transaction.get("status") != "prepared":
            return PlainTextResponse("Authorization state is unknown or expired. Restart consent.", status_code=400)
        try:
            code = validate_callback(
                expected_state=transaction.get("state"),
                returned_state=returned_state,
                code=request.query_params.get("code"),
                used=False,
            )
        except IBKROAuthCallbackError:
            return PlainTextResponse("Authorization callback was invalid. Restart consent.", status_code=400)

        try:
            created_at = datetime.fromisoformat(transaction["created_at"])
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age < 0 or age > config.callback_timeout_seconds:
                return PlainTextResponse("Authorization state has expired. Restart consent.", status_code=400)
            run_id = transaction["run_id"]
            setup_only = run_id == "ibkr-integration-setup"
            if not setup_only and not app.state.workflow_lifecycle.is_authorization_required(run_id):
                return PlainTextResponse("The workflow is no longer waiting for IBKR authorization.", status_code=409)
            transaction["status"] = "processing"
            await asyncio.to_thread(store.persist_oauth_transaction, transaction)
            exchanger = app.state.code_exchanger or exchange_authorization_code
            tokens = await exchanger(store, config, transaction, code)
            _require_ibkr_read_scope(tokens.get("scope"))
            await asyncio.to_thread(store.persist_oauth_transaction, {
                "status": "completed", "state": "consumed", "code_verifier": "consumed",
                "run_id": run_id, "created_at": transaction["created_at"],
            })
            if not setup_only:
                await asyncio.to_thread(app.state.workflow_lifecycle.resume, run_id)
        except (IBKRScopeRejected, IBKRStorageUnavailable):
            return PlainTextResponse("IBKR did not grant the exact mcp.read scope. Restart consent after review.", status_code=403)
        except Exception as exc:
            types, statuses = _safe_failure_tree(exc)
            _LOGGER.warning(
                "Harness hosted OAuth callback failed stage=token_exchange nested_exception_types=%s http_status=%s",
                ",".join(types), ",".join(str(value) for value in statuses) or "none",
            )
            return PlainTextResponse("IBKR authorization could not be completed. Restart consent after review.", status_code=502)
        if run_id == "ibkr-integration-setup":
            return PlainTextResponse("IBKR mcp.read authorization completed and stored by Harness. No workflow or MPF tools were enabled.")
        return PlainTextResponse("IBKR mcp.read authorization completed. The Harness workflow resumed with fresh data. No MPF tools were enabled.")

    async def prepare_authorization(run_id: str, diagnostics: dict[str, Any]) -> str:
        assert config is not None
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        storage = HarnessMCPTokenStorage(app.state.store)
        metadata = OAuthClientMetadata(
            redirect_uris=[config.redirect_uri], application_type="web",
            token_endpoint_auth_method="none", scope=READ_ONLY_SCOPE,
            client_name="Agentic Harness read-only MCP",
        )

        class HostedProvider(OAuthClientProvider):
            async def _perform_authorization_code_grant(self) -> tuple[str, str]:
                authorization_url = await create_hosted_authorization_link(
                    self.context, config=config, run_id=run_id,
                    transaction_store=app.state.store,
                )
                await hosted_redirect(authorization_url)
                raise _AuthorizationLinkReady()

        async def hosted_redirect(url: str) -> None:
            parts = urlsplit(url)
            states = parse_qs(parts.query).get("state", [])
            if parts.scheme != "https" or parts.hostname != _IBKR_HOST or len(states) != 1 or not states[0]:
                raise IBKROAuthConfigurationError("IBKR returned an unexpected authorization destination")
            app.state.prepared_authorization_url = url

        auth = HostedProvider(
            server_url=IBKR_MCP_SERVER_URL, client_metadata=metadata, storage=storage,
            redirect_handler=hosted_redirect, callback_handler=None,
            allowed_scopes={READ_ONLY_SCOPE},
        )
        diagnostics.update({"requested_scope": READ_ONLY_SCOPE, "application_type": "web"})
        diagnostics["callback_uri"] = safe_endpoint_identity(config.redirect_uri)
        stage = "oauth_metadata_discovery"
        registration_endpoint = await discover_ibkr_oauth_metadata(auth)
        diagnostics["registration_endpoint"] = safe_endpoint_identity(registration_endpoint)
        remote = RemoteMCPClient(RemoteMCPProfile(IBKR_MCP_SERVER_URL, READ_ONLY_SCOPE))
        # The MCP SDK builds and sends DCR from the discovered metadata. The
        # request hook pins the exact advertised endpoint and exact web/read body.
        async with httpx.AsyncClient(
            auth=auth,
            timeout=httpx.Timeout(30, read=45),
            event_hooks=oauth_diagnostic_hooks(auth, registration_endpoint=registration_endpoint, diagnostics=diagnostics),
        ) as http_client:
            # RemoteMCPClient owns the MCP SDK session; use the configured
            # authenticated HTTP client so its safe DCR hook observes the POST.
            async with remote.connect(auth, http_client=http_client) as _:
                pass
        raise IBKROAuthConfigurationError("IBKR authorization completed without a hosted redirect")

    async def exchange_authorization_code(
        store: SecretManagerIBKRCredentialStore,
        config: HarnessIBKROAuthConfig,
        transaction: dict[str, Any],
        code: str,
    ) -> dict[str, Any]:
        """Complete a callback on any instance using the SDK token primitives."""
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        storage = HarnessMCPTokenStorage(store)
        metadata = OAuthClientMetadata(
            redirect_uris=[config.redirect_uri], application_type="web",
            token_endpoint_auth_method="none", scope=READ_ONLY_SCOPE,
            client_name="Agentic Harness read-only MCP",
        )
        auth = OAuthClientProvider(
            server_url=IBKR_MCP_SERVER_URL, client_metadata=metadata,
            storage=storage, allowed_scopes={READ_ONLY_SCOPE},
        )
        await auth._initialize()
        await discover_ibkr_oauth_metadata(auth)
        _require_ibkr_read_scope(auth.context.client_metadata.scope)
        request = await auth._exchange_token_authorization_code(code, transaction["code_verifier"])
        token_endpoint = _safe_ibkr_url(auth.context.oauth_metadata.token_endpoint, "token endpoint")
        if request.url != token_endpoint:
            raise IBKROAuthConfigurationError("IBKR token request did not use the discovered endpoint")
        async with httpx.AsyncClient(timeout=httpx.Timeout(20, read=30)) as client:
            response = await client.send(request)
            await response.aread()
            if response.status_code != 200:
                raise IBKROAuthConfigurationError("IBKR authorization code exchange failed")
            payload = json.loads(response.content)
            if not isinstance(payload, dict):
                raise IBKRScopeRejected("IBKR token response did not contain an exact mcp.read grant")
            _require_ibkr_read_scope(payload.get("scope"))
            await auth._handle_token_response(response)
        tokens = await storage.get_tokens()
        if tokens is None:
            raise IBKRStorageUnavailable("IBKR token response was not persisted")
        result = tokens.model_dump(mode="json", exclude_none=True)
        _require_exact_read_scope(IBKRCredentials(client_info={}, tokens=result))
        return result

    app.routes.extend([
        Route("/health", health, methods=["GET"]),
        Route("/ibkr/reconnect", begin_reconnect, methods=["POST"]),
        Route(CALLBACK_PATH, receive_callback, methods=["GET"]),
    ])
    return app


def run_oauth_sync(coro: Any) -> Any:
    """Run SDK refresh work from Harness's synchronous tool handler."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class HarnessIBKRMCPDataProvider:
    """OAuth-capable provider shell; data actions remain unmapped pending schema review."""

    mapping_verified = False

    def __init__(self, credential_store: SecretManagerIBKRCredentialStore, *, redirect_uri: str) -> None:
        self.credential_store = credential_store
        self.redirect_uri = redirect_uri

    def refresh_credentials(self, credentials: IBKRCredentials) -> IBKRCredentials:
        _require_exact_read_scope(credentials)
        if not credentials.tokens.get("refresh_token"):
            raise IBKRAuthorizationRequired("interactive consent is required")
        try:
            result = run_oauth_sync(self._refresh_tokens())
        except (IBKRScopeRejected, IBKRStorageUnavailable):
            raise
        except Exception as exc:
            raise IBKRAuthorizationRequired("interactive consent is required") from exc
        if result is None:
            raise IBKRAuthorizationRequired("interactive consent is required")
        _require_exact_read_scope(result)
        return result

    async def _refresh_tokens(self, *, transport: httpx.AsyncBaseTransport | None = None) -> IBKRCredentials | None:
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        storage = HarnessMCPTokenStorage(self.credential_store)
        metadata = OAuthClientMetadata(
            redirect_uris=[self.redirect_uri],
            application_type="web",
            token_endpoint_auth_method="none",
            scope=READ_ONLY_SCOPE,
            client_name="Agentic Harness read-only MCP",
        )
        auth = OAuthClientProvider(
            server_url=IBKR_MCP_SERVER_URL,
            client_metadata=metadata,
            storage=storage,
            allowed_scopes={READ_ONLY_SCOPE},
        )
        await auth._initialize()
        stored_tokens = await asyncio.to_thread(self.credential_store.load_tokens)
        if stored_tokens and stored_tokens.get("expires_in") is not None:
            issued_at = stored_tokens.get("harness_obtained_at")
            try:
                if not isinstance(issued_at, str) or not issued_at:
                    raise ValueError("missing token issue time")
                issued_timestamp = datetime.fromisoformat(issued_at).timestamp()
                auth.context.token_expiry_time = issued_timestamp + float(stored_tokens["expires_in"])
            except (TypeError, ValueError, OverflowError):
                auth.context.token_expiry_time = 1.0
        elif stored_tokens and stored_tokens.get("refresh_token"):
            # Without expiry provenance, do not treat a persisted access token
            # as valid forever. Attempt the supported refresh grant.
            auth.context.token_expiry_time = 1.0
        token = auth.context.current_tokens
        if token is None:
            return None
        if auth.context.is_token_valid():
            client_info = auth.context.client_info
            if client_info is None:
                return None
            if stored_tokens is None:
                return None
            return IBKRCredentials(
                client_info=client_info.model_dump(mode="json", exclude_none=True),
                tokens=stored_tokens,
            )
        if not token.refresh_token or not auth.context.can_refresh_token():
            return None
        # Discovery is needed to use only the token endpoint advertised by IBKR.
        await discover_ibkr_oauth_metadata(auth, transport=transport)
        request = await auth._refresh_token()
        token_endpoint = _safe_ibkr_url(auth.context.oauth_metadata.token_endpoint, "token endpoint")
        if request.url != token_endpoint:
            raise IBKROAuthConfigurationError("IBKR refresh did not use the discovered token endpoint")
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(20, read=30)) as client:
            response = await client.send(request)
            if response.status_code != 200:
                return None
            await response.aread()
            payload = json.loads(response.content)
            if not isinstance(payload, dict):
                raise IBKRScopeRejected("IBKR refresh response did not contain an exact mcp.read grant")
            _require_ibkr_read_scope(payload.get("scope"))
            if not await auth._handle_refresh_response(response):
                return None
        saved = await storage.get_tokens()
        client_info = await storage.get_client_info()
        persisted_tokens = await asyncio.to_thread(self.credential_store.load_tokens)
        if saved is None or client_info is None or persisted_tokens is None:
            return None
        result = IBKRCredentials(
            client_info=client_info.model_dump(mode="json", exclude_none=True),
            tokens=persisted_tokens,
        )
        _require_exact_read_scope(result)
        return result

    def invoke(self, action: str, arguments: dict[str, Any], credentials: IBKRCredentials) -> Any:
        _require_exact_read_scope(credentials)
        raise IBKRProviderMappingUnavailable("Authenticated IBKR tool schemas have not been reviewed")


def main() -> None:
    import uvicorn

    app = create_ibkr_oauth_app()
    port = int(os.environ.get("PORT", "8080"))
    # Never emit callback query parameters or response bodies in application logs.
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
