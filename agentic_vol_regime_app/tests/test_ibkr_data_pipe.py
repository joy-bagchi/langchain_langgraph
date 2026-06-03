from __future__ import annotations

from pathlib import Path

from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKROptionChainRequest,
)
from agentic_vol_regime_app.data.market_data_loader import load_market_snapshot


class FakeIBKRClient:
    def __init__(self, _connection: IBKRConnectionConfig) -> None:
        self.connection = _connection

    def fetch_market_snapshot(self, request: IBKROptionChainRequest) -> dict:
        return {
            "schema_version": "observation.v1",
            "as_of": "2026-06-02T20:00:00Z",
            "source": "IBKR",
            "symbols": {
                request.symbol: {
                    "last": 599.12,
                    "close": 598.44,
                    "bid": 599.1,
                    "ask": 599.14,
                    "volume": 81234000,
                },
                "VIX": {"last": 17.8},
                "VVIX": {"last": 95.2},
                "VIX9D": {"last": 17.1},
                "VIX3M": {"last": 19.3},
            },
            "history": {},
            "quality": {"is_complete": True, "warnings": [], "stale_fields": []},
            "option_chain": {
                "underlying_symbol": request.symbol,
                "underlying_price": 599.12,
                "fetched_at": "2026-06-02T20:00:00Z",
                "exchange": request.option_exchange,
                "currency": request.currency,
                "expirations": ["20260620", "20260718"],
                "strikes": [590.0, 600.0],
                "rights": ["C", "P"],
                "option_quotes": [
                    {
                        "symbol": "SPY   260620C00600000",
                        "expiry": "20260620",
                        "strike": 600.0,
                        "right": "C",
                        "exchange": "SMART",
                        "currency": "USD",
                        "bid": 9.8,
                        "ask": 10.1,
                        "last": 9.95,
                        "close": 9.72,
                        "mark": 9.95,
                        "volume": 1520,
                        "open_interest": 11234,
                        "bid_size": 12,
                        "ask_size": 15,
                        "last_size": 4,
                        "multiplier": "100",
                        "greeks": {
                            "delta": 0.47,
                            "gamma": 0.03,
                            "theta": -0.18,
                            "vega": 0.11,
                            "implied_vol": 0.164,
                            "opt_price": 9.95,
                            "pv_dividend": 0.0,
                            "und_price": 599.12,
                        },
                    }
                ],
            },
            "provider_metadata": {
                "host": self.connection.host,
                "port": self.connection.port,
                "client_id": self.connection.client_id,
            },
        }


def test_ibkr_data_pipe_normalizes_option_chain_snapshot() -> None:
    pipe = IBKRDataPipe(
        connection=IBKRConnectionConfig(host="127.0.0.1", port=7497, client_id=99),
        client_factory=FakeIBKRClient,
    )
    observation = pipe.fetch_market_snapshot(
        IBKROptionChainRequest(symbol="SPY", expiry_count=2, strike_count=2)
    )

    assert observation.source == "IBKR"
    assert observation.symbols["SPY"]["last"] == 599.12
    assert observation.option_chain["underlying_symbol"] == "SPY"
    assert len(observation.option_chain["option_quotes"]) == 1
    assert observation.option_chain["option_quotes"][0]["greeks"]["delta"] == 0.47


def test_market_data_loader_supports_ibkr_provider_payload() -> None:
    pipe = IBKRDataPipe(
        connection=IBKRConnectionConfig(client_id=77),
        client_factory=FakeIBKRClient,
    )
    observation = load_market_snapshot(
        {
            "data_provider": "ibkr",
            "symbol": "SPY",
            "ibkr": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 77,
                "expiry_count": 2,
                "strike_count": 2,
            },
        },
        app_root=Path.cwd(),
        data_pipe=pipe,
    )

    assert observation.source == "IBKR"
    assert observation.provider_metadata["client_id"] == 77
    assert observation.option_chain["expirations"] == ["20260620", "20260718"]
