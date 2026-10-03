from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.agentic_os.tool_service import ToolExecutionRequest
from agentic_harness.contracts import WorkflowDefinition, WorkflowStep
from agentic_harness.ibkr_data_reader import (
    IBKRAuthorizationRequired,
    IBKRCredentials,
    ProviderResult,
    READ_ONLY_SCOPE,
    SecretManagerIBKRCredentialStore,
    IBKRScopeRejected,
)
from agentic_harness.notifications import WorkflowNotification
from agentic_harness.runtime import WorkflowRunner, inspect_run


TOKEN = "fake-access-token-must-never-leak"


class FakeCredentialStore:
    def __init__(self, credentials: IBKRCredentials | None) -> None:
        self.credentials = credentials
        self.persisted: list[IBKRCredentials] = []

    def load(self) -> IBKRCredentials | None:
        return self.credentials

    def persist(self, credentials: IBKRCredentials) -> None:
        self.persisted.append(credentials)
        self.credentials = credentials


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.daily_calls = 0
        self.require_consent_for: set[str] = set()
        self.refresh_requires_consent = False

    def refresh_credentials(self, credentials: IBKRCredentials) -> IBKRCredentials:
        if self.refresh_requires_consent:
            raise IBKRAuthorizationRequired("provider details must remain private")
        return IBKRCredentials(
            client_info=credentials.client_info,
            tokens={"scope": READ_ONLY_SCOPE, "access_token": "rotated-token"},
        )

    def invoke(self, action: str, arguments: dict, credentials: IBKRCredentials) -> ProviderResult:
        assert credentials.scope == READ_ONLY_SCOPE
        self.calls.append((action, dict(arguments)))
        if action in self.require_consent_for:
            raise IBKRAuthorizationRequired("safe marker")
        if action == "get_symbol_daily_data":
            self.daily_calls += 1
            return ProviderResult({
                "symbol": arguments["symbol"], "trading_date": arguments["trading_date"],
                "fields": {"open": 100.0, "high": 105.0, "low": 99.0,
                           "close": 103.0 + self.daily_calls, "volume": 1200},
                "units": {"open": "USD/share", "volume": "shares", "provider_secret": TOKEN},
                "source": "untrusted-source-field",
                "observation_time": "2026-10-01T20:00:00Z",
                "quality": "live",
                "access_token": TOKEN,
                "raw_provider_catalog": [{"name": "place_order"}],
            })
        if action == "list_option_contracts":
            return ProviderResult({
                "contracts": [
                    {"exact_contract_id": "conid-7", "right": "C", "strike": 600,
                     "expiry": "20261016", "provider_secret": TOKEN},
                    {"exact_contract_id": "conid-8", "right": "P", "strike": 590,
                     "expiry": "20261016"},
                ],
                "pagination": {"next_cursor": "page-2", "has_more": True, "access_token": TOKEN},
                "quality": "delayed",
                "observation_time": "2026-10-01T20:00:00Z",
            })
        return ProviderResult({
            "exact_contract_id": arguments.get("exact_contract_id", "conid-7"),
            "symbol": arguments.get("symbol", "SPY"),
            "right": arguments.get("right", "C"),
            "strike": arguments.get("strike", 600),
            "expiry": arguments.get("expiry", "20261016"),
            "fields": {"price": 11.25, "bid": 11.1, "ask": 11.4, "iv": None, "volume": 1200},
            "units": {"price": "USD/contract", "iv": "fraction", "provider_secret": TOKEN},
            "quality": "delayed",
            "observation_time": "2026-10-01T20:00:00Z",
            "provider_error_detail": TOKEN,
        })


def credentials(scope: str = READ_ONLY_SCOPE) -> IBKRCredentials:
    return IBKRCredentials(
        client_info={"client_id": "fake-client"},
        tokens={"scope": scope, "access_token": TOKEN, "refresh_token": "fake-refresh-token"},
    )


def test_registry_exposes_only_the_three_bounded_ibkr_actions(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = FakeCredentialStore(credentials())
    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=store,
    )
    tools = {tool.tool_id: tool for tool in services.tools.list_tools()}
    actions = {name for name, definition in tools.items() if definition.metadata.get("tool_type") == "ibkr_data_reader"}

    assert actions == {"get_symbol_daily_data", "list_option_contracts", "get_option_data"}
    assert "ibkr_data_pipeline" not in tools
    assert all(tools[action].input_schema.get("additionalProperties") is False for action in actions)
    assert "place_order" not in str([tools[action].input_schema for action in actions])


def test_daily_data_is_projected_and_rotated_credentials_stay_private(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = FakeCredentialStore(credentials())
    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=store,
    )

    response = services.tools.execute(ToolExecutionRequest(
        tool_id="get_symbol_daily_data",
        arguments={"symbol": "SPY", "trading_date": "2026-10-01"},
        metadata={"run_id": "run-1"},
    ))
    rendered = repr(response)

    assert response.status == "succeeded"
    assert response.output["data_status"] == "found"
    assert response.output["source"] == "IBKR"
    assert response.output["fields"]["close"] == 104.0
    assert response.output["unavailable_fields"] == []
    assert response.output["quality"] == "live"
    assert response.output["retrieval_time"]
    assert response.output["observation_time"] == "2026-10-01T20:00:00+00:00"
    assert response.output["units"] == {"open": "USD/share", "volume": "shares"}
    assert len(store.persisted) == 1
    assert TOKEN not in rendered
    assert TOKEN not in repr(provider.calls)


def test_contract_listing_and_option_data_are_exact_and_explicit(tmp_path: Path) -> None:
    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=FakeProvider(),
        ibkr_credential_store=FakeCredentialStore(credentials()),
    )
    listed = services.tools.execute(ToolExecutionRequest(
        "list_option_contracts",
        {"symbol": "SPY", "optional_expiry": "20261016", "pagination": {"limit": 20}},
    ))
    option = services.tools.execute(ToolExecutionRequest(
        "get_option_data",
        {"exact_contract_id": "conid-7"},
    ))
    assert listed.status == "succeeded"
    assert listed.output["contracts"] == [
        {"exact_contract_id": "conid-7", "right": "C", "strike": 600, "expiry": "20261016",
         "available_fields": ["exact_contract_id", "right", "strike", "expiry"], "unavailable_fields": []},
        {"exact_contract_id": "conid-8", "right": "P", "strike": 590, "expiry": "20261016",
         "available_fields": ["exact_contract_id", "right", "strike", "expiry"], "unavailable_fields": []},
    ]
    assert listed.output["pagination"] == {"next_cursor": "page-2", "has_more": True}
    assert option.output["contract_status"] == "found"
    assert option.output["fields"] == {"price": 11.25, "bid": 11.1, "ask": 11.4, "iv": None, "volume": 1200}
    assert option.output["unavailable_fields"] == ["iv"]
    assert option.output["quality"] == "delayed"


def test_missing_option_contract_does_not_substitute_an_alternate(tmp_path: Path) -> None:
    class MissingProvider(FakeProvider):
        def invoke(self, action, arguments, credentials):
            return ProviderResult({"contract_status": "missing", "quality": "unavailable"})

    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=MissingProvider(),
        ibkr_credential_store=FakeCredentialStore(credentials()),
    )
    response = services.tools.execute(ToolExecutionRequest(
        "get_option_data",
        {"symbol": "SPY", "right": "C", "strike": 600, "expiry": "20261016"},
    ))
    assert response.status == "succeeded"
    assert response.output["contract_status"] == "missing"
    assert response.output["unavailable_fields"] == ["price", "bid", "ask", "iv", "volume"]
    assert response.output["fields"] == {"price": None, "bid": None, "ask": None, "iv": None, "volume": None}
    assert response.output["requested_contract"] == {
        "symbol": "SPY", "right": "C", "strike": 600, "expiry": "20261016"
    }


def test_daily_date_and_option_expiry_mismatches_are_not_substituted(tmp_path: Path) -> None:
    class MismatchProvider(FakeProvider):
        def invoke(self, action, arguments, credentials):
            if action == "get_symbol_daily_data":
                return ProviderResult({
                    "symbol": "SPY", "trading_date": "2026-10-02",
                    "fields": {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                    "quality": "live",
                })
            return ProviderResult({
                "contracts": [{"exact_contract_id": "wrong-expiry", "right": "C", "strike": 600,
                               "expiry": "20261023"}],
                "quality": "delayed",
            })

    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=MismatchProvider(),
        ibkr_credential_store=FakeCredentialStore(credentials()),
    )
    daily = services.tools.execute(ToolExecutionRequest(
        "get_symbol_daily_data", {"symbol": "SPY", "trading_date": "2026-10-01"},
    ))
    contracts = services.tools.execute(ToolExecutionRequest(
        "list_option_contracts", {"symbol": "SPY", "optional_expiry": "20261016"},
    ))
    assert daily.output["data_status"] == "unverified"
    assert daily.output["fields"] == {"open": None, "high": None, "low": None, "close": None, "volume": None}
    assert contracts.output["contract_status"] == "missing"
    assert contracts.output["contracts"] == []


def test_broader_scope_is_rejected_and_provider_is_not_called(tmp_path: Path) -> None:
    provider = FakeProvider()
    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=FakeCredentialStore(credentials("mcp.read mcp.write")),
    )
    response = services.tools.execute(ToolExecutionRequest(
        "get_symbol_daily_data", {"symbol": "SPY", "trading_date": "2026-10-01"},
    ))
    assert response.status == "scope_rejected"
    assert provider.calls == []
    assert "mcp.write" not in repr(response)


def test_secret_manager_store_reads_and_versions_only_exact_read_grants() -> None:
    class FakeSecretManager:
        def __init__(self) -> None:
            self.values: dict[str, bytes] = {}
            self.added: list[tuple[str, bytes]] = []

        def access_secret_version(self, *, request: dict) -> object:
            secret = request["name"].split("/secrets/")[1].split("/")[0]
            if secret not in self.values:
                class NotFound(Exception):
                    pass
                raise NotFound()
            return SimpleNamespace(payload=SimpleNamespace(data=self.values[secret]))

        def add_secret_version(self, *, request: dict) -> None:
            secret = request["parent"].split("/secrets/")[1]
            value = bytes(request["payload"]["data"])
            self.values[secret] = value
            self.added.append((secret, value))

    client = FakeSecretManager()
    store = SecretManagerIBKRCredentialStore(
        project_id="test-project", client_secret_id="ibkr-client", token_secret_id="ibkr-token", client=client,
    )
    grant = credentials()
    store.persist(grant)
    loaded = store.load()
    assert loaded is not None
    assert loaded.scope == READ_ONLY_SCOPE
    rotated = IBKRCredentials(
        client_info=grant.client_info,
        tokens={"scope": READ_ONLY_SCOPE, "access_token": "new-token"},
    )
    store.persist(rotated)
    assert [secret for secret, _ in client.added] == ["ibkr-client", "ibkr-token", "ibkr-token"]
    assert json.loads(client.values["ibkr-token"].decode())["access_token"] == "new-token"
    try:
        store.persist(credentials("mcp.read mcp.orders.submit"))
    except IBKRScopeRejected:
        pass
    else:
        raise AssertionError("broader grant was accepted")
    assert len(client.added) == 3


def test_client_registration_with_write_scope_is_rejected(tmp_path: Path) -> None:
    bad = IBKRCredentials(
        client_info={"client_id": "fake", "scope": "mcp.read mcp.write"},
        tokens={"scope": READ_ONLY_SCOPE, "access_token": TOKEN},
    )
    provider = FakeProvider()
    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=FakeCredentialStore(bad),
    )
    response = services.tools.execute(ToolExecutionRequest(
        "get_symbol_daily_data", {"symbol": "SPY", "trading_date": "2026-10-01"},
    ))
    assert response.status == "scope_rejected"
    assert provider.calls == []


def test_provider_error_details_are_not_returned(tmp_path: Path) -> None:
    class ErrorProvider(FakeProvider):
        def invoke(self, action, arguments, credentials):
            raise RuntimeError(f"provider response contained {TOKEN}")

    services = build_platform_services(
        storage_root=tmp_path,
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=ErrorProvider(),
        ibkr_credential_store=FakeCredentialStore(credentials()),
    )
    response = services.tools.execute(ToolExecutionRequest(
        "get_symbol_daily_data", {"symbol": "SPY", "trading_date": "2026-10-01"},
    ))
    assert response.status == "unavailable"
    assert response.metadata["reason"] == "provider_unavailable"
    assert TOKEN not in repr(response)


def test_expired_grant_checkpoints_until_refresh_is_authorized(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.refresh_requires_consent = True
    notifications: list[WorkflowNotification] = []
    services = build_platform_services(
        storage_root=tmp_path / "runtime",
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=FakeCredentialStore(credentials()),
        notification_handler=notifications.append,
    )
    definition = WorkflowDefinition(
        workflow_id="expired_ibkr_grant",
        title="Expired IBKR grant",
        entry_step="daily",
        memory_namespace="tests",
        steps={
            "daily": WorkflowStep(
                step_id="daily", title="Daily", step_type="tool", output_key="daily_data",
                metadata={"tool_id": "get_symbol_daily_data",
                          "arguments": {"symbol": "SPY", "trading_date": "2026-10-01"}},
            ),
        },
    )
    runner = WorkflowRunner(definition, storage_root=tmp_path / "runtime", services=services)
    paused = runner.start({}, run_id="expired-grant", initial_state_overrides={
        "allowed_tools": ["get_symbol_daily_data"],
    })

    assert paused["status"] == "authorization_required"
    assert inspect_run("expired-grant", storage_root=tmp_path / "runtime")["status"] == "authorization_required"
    assert provider.calls == []
    assert len(notifications) == 1

    provider.refresh_requires_consent = False
    resumed = runner.resume("expired-grant")
    assert resumed["status"] == "completed"
    assert provider.daily_calls == 1


def test_consent_checkpoint_resumes_from_a_fresh_market_data_step(tmp_path: Path) -> None:
    provider = FakeProvider()
    provider.require_consent_for.add("get_option_data")
    store = FakeCredentialStore(credentials())
    notifications: list[WorkflowNotification] = []
    services = build_platform_services(
        storage_root=tmp_path / "runtime",
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=store,
        notification_handler=notifications.append,
    )
    definition = WorkflowDefinition(
        workflow_id="ibkr_authorization_resume",
        title="IBKR authorization resume",
        entry_step="daily",
        memory_namespace="tests",
        workflow_path=None,
        steps={
            "daily": WorkflowStep(
                step_id="daily", title="Daily", step_type="tool", output_key="daily_data",
                next_step="option", metadata={"tool_id": "get_symbol_daily_data",
                                               "arguments": {"symbol": "SPY", "trading_date": "2026-10-01"}},
            ),
            "option": WorkflowStep(
                step_id="option", title="Option", step_type="tool", output_key="option_data",
                metadata={"tool_id": "get_option_data", "arguments": {"exact_contract_id": "conid-7"}},
            ),
        },
    )
    runner = WorkflowRunner(definition, storage_root=tmp_path / "runtime", services=services)
    paused = runner.start({}, run_id="ibkr-run", initial_state_overrides={
        "allowed_tools": ["get_symbol_daily_data", "get_option_data"],
        "named_outputs": {"daily_data": {"fields": {"close": -1}}},
    })
    inspected = inspect_run("ibkr-run", storage_root=tmp_path / "runtime")

    assert paused["status"] == "authorization_required"
    assert paused["pending_authorization"]["fresh_market_data_required"] is True
    assert paused["pending_authorization"]["reconnect_action"]["route"] == "/ibkr/reconnect"
    assert paused["pending_authorization"]["reconnect_action"]["requires_authenticated_harness_session"] is True
    assert inspected["status"] == "authorization_required"
    assert any(event.get("type") == "authorization_required" for event in inspected["events"])
    assert len(notifications) == 1
    assert notifications[0].notification_type == "ibkr_authorization_required"
    assert notifications[0].metadata["action"]["route"] == "/ibkr/reconnect"
    assert "daily_data" in paused["named_outputs"]

    provider.require_consent_for.clear()  # simulate verified read-only consent completion
    resumed = runner.resume("ibkr-run")

    assert resumed["status"] == "completed"
    assert resumed["pending_authorization"] is None
    assert resumed["named_outputs"]["daily_data"]["fields"]["close"] == 105.0
    assert provider.daily_calls == 2
    assert [action for action, _ in provider.calls] == [
        "get_symbol_daily_data", "get_option_data", "get_symbol_daily_data", "get_option_data"
    ]
    persisted = inspect_run("ibkr-run", storage_root=tmp_path / "runtime")
    assert TOKEN not in repr(persisted)
