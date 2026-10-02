from __future__ import annotations

from pathlib import Path

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.ibkr_data_reader import IBKRCredentials, ProviderResult, READ_ONLY_SCOPE
from agentic_harness.runtime import run_agent_workflow


class FakeCredentialStore:
    def __init__(self) -> None:
        self.credentials = IBKRCredentials(
            client_info={"client_id": "fake"},
            tokens={"scope": READ_ONLY_SCOPE, "access_token": "fake-token"},
        )

    def load(self):
        return self.credentials

    def persist(self, credentials):
        self.credentials = credentials


class FakeIBKRProvider:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def refresh_credentials(self, credentials):
        return credentials

    def invoke(self, action, arguments, credentials):
        self.actions.append(action)
        if action == "get_symbol_daily_data":
            return ProviderResult({
                "symbol": arguments["symbol"], "trading_date": arguments["trading_date"],
                "fields": {"open": 602.1, "high": 604.0, "low": 601.4, "close": 603.12, "volume": 71234000},
                "units": {"open": "USD/share", "volume": "shares"},
                "observation_time": "2026-06-03T20:00:00Z",
                "quality": "live",
            })
        if action == "list_option_contracts":
            return ProviderResult({
                "contracts": [{"exact_contract_id": "conid-600c", "right": "C", "strike": 600,
                               "expiry": "20260620"}],
                "pagination": {"has_more": False},
                "observation_time": "2026-06-03T20:00:00Z",
                "quality": "delayed",
            })
        return ProviderResult({
            "exact_contract_id": arguments["exact_contract_id"],
            "fields": {"price": 11.25, "bid": 11.1, "ask": 11.4, "iv": 0.171, "volume": 1620},
            "units": {"price": "USD/contract", "iv": "fraction", "volume": "contracts"},
            "observation_time": "2026-06-03T20:00:00Z",
            "quality": "delayed",
        })


def test_ibkr_market_data_agent_uses_only_the_bounded_harness_actions(tmp_path: Path) -> None:
    provider = FakeIBKRProvider()
    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        langsmith_tracing=False,
        ibkr_data_reader_provider=provider,
        ibkr_credential_store=FakeCredentialStore(),
    )

    result = run_agent_workflow(
        Path("agentic_vol_regime_app/configs/agents/ibkr_market_data_agent.yaml"),
        {
            "symbol": "SPY",
            "trading_date": "2026-06-03",
            "exact_contract_id": "conid-600c",
        },
        storage_root=tmp_path / ".workflow_memory",
        services=services,
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["daily_data"]["fields"]["close"] == 603.12
    assert result["named_outputs"]["option_contracts"]["contracts"][0]["exact_contract_id"] == "conid-600c"
    assert result["named_outputs"]["option_data"]["fields"]["iv"] == 0.171
    assert result["agent"]["allowed_tools"] == [
        "get_symbol_daily_data", "list_option_contracts", "get_option_data"
    ]
    assert provider.actions == ["get_symbol_daily_data", "list_option_contracts", "get_option_data"]
