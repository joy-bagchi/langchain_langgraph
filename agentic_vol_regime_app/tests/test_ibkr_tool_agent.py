from __future__ import annotations

from pathlib import Path

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.runtime import run_agent_workflow


class FakeIBKRPipe:
    def fetch_market_snapshot(self, request) -> object:
        class Snapshot:
            def to_dict(self_inner) -> dict:
                return {
                    "schema_version": "observation.v1",
                    "as_of": "2026-06-03T20:00:00Z",
                    "source": "IBKR",
                    "symbols": {
                        request.symbol: {
                            "last": 603.12,
                            "close": 602.44,
                            "bid": 603.1,
                            "ask": 603.2,
                            "volume": 71234000,
                        }
                    },
                    "history": {},
                    "quality": {"is_complete": True, "warnings": [], "stale_fields": []},
                    "option_chain": {
                        "underlying_symbol": request.symbol,
                        "underlying_price": 603.12,
                        "fetched_at": "2026-06-03T20:00:00Z",
                        "exchange": "SMART",
                        "currency": "USD",
                        "expirations": ["20260620"],
                        "strikes": [600.0],
                        "rights": ["C", "P"],
                        "option_quotes": [
                            {
                                "symbol": "SPY   260620C00600000",
                                "expiry": "20260620",
                                "strike": 600.0,
                                "right": "C",
                                "exchange": "SMART",
                                "currency": "USD",
                                "bid": 11.1,
                                "ask": 11.4,
                                "last": 11.25,
                                "close": 10.9,
                                "mark": 11.25,
                                "volume": 1620,
                                "open_interest": 13234,
                                "bid_size": 12,
                                "ask_size": 15,
                                "last_size": 4,
                                "multiplier": "100",
                                "greeks": {
                                    "delta": 0.51,
                                    "gamma": 0.03,
                                    "theta": -0.16,
                                    "vega": 0.12,
                                    "implied_vol": 0.171,
                                },
                            }
                        ],
                    },
                    "provider_metadata": {"port": 4001},
                }

        return Snapshot()


def test_ibkr_market_data_agent_uses_ibkr_tool(tmp_path: Path) -> None:
    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        ibkr_data_pipe=FakeIBKRPipe(),
    )

    result = run_agent_workflow(
        Path("agentic_vol_regime_app/configs/agents/ibkr_market_data_agent.yaml"),
        {
            "symbol": "SPY",
            "host": "127.0.0.1",
            "port": 4001,
            "client_id": 73,
            "market_data_type": 1,
            "exchange": "SMART",
            "option_exchange": "SMART",
            "currency": "USD",
            "expiry_count": 2,
            "strike_count": 8,
            "min_days_to_expiry": 0,
        },
        storage_root=tmp_path / ".workflow_memory",
        services=services,
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["requested_symbol"] == "SPY"
    assert result["named_outputs"]["ibkr_snapshot"]["source"] == "IBKR"
    assert result["named_outputs"]["ibkr_snapshot"]["provider_metadata"]["port"] == 4001
    assert result["named_outputs"]["ibkr_snapshot"]["option_chain"]["option_quotes"][0]["greeks"]["delta"] == 0.51
    assert result["agent"]["allowed_tools"] == ["ibkr_data_pipeline"]
