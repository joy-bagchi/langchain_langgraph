from __future__ import annotations

import asyncio
from pathlib import Path

from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKRLiveClient,
    IBKROptionChainRequest,
    IBKRVolRegimeSnapshotRequest,
    _ensure_thread_event_loop,
)
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.quality import validate_observation
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

    def fetch_vol_regime_snapshot(self, request: IBKRVolRegimeSnapshotRequest) -> dict:
        option_request = request.option_chain
        return {
            "schema_version": "observation.v1",
            "as_of": "2026-06-02T20:00:00Z",
            "source": "IBKR",
            "symbols": {
                option_request.symbol: {
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
            "history": {
                "SPY_close": [580.0 + index for index in range(30)],
                "VIX": [16.0 + (index * 0.08) for index in range(30)],
                "VVIX": [92.0 + (index * 0.25) for index in range(30)],
                "VIX9D": [15.8 + (index * 0.07) for index in range(30)],
                "VIX3M": [19.8 + (index * 0.03) for index in range(30)],
            },
            "quality": {"is_complete": True, "warnings": [], "stale_fields": []},
            "option_chain": self.fetch_market_snapshot(option_request)["option_chain"],
            "provider_metadata": {
                "host": self.connection.host,
                "port": self.connection.port,
                "client_id": self.connection.client_id,
                "history_days": request.history_days,
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


def test_ibkr_data_pipe_normalizes_vol_regime_snapshot() -> None:
    pipe = IBKRDataPipe(
        connection=IBKRConnectionConfig(host="127.0.0.1", port=4001, client_id=73),
        client_factory=FakeIBKRClient,
    )

    observation = pipe.fetch_vol_regime_snapshot(
        IBKRVolRegimeSnapshotRequest.from_payload(
            {
                "symbol": "SPY",
                "port": 4001,
                "history_days": 30,
                "expiry_count": 2,
                "strike_count": 2,
            }
        )
    )

    assert observation.source == "IBKR"
    assert observation.symbols["VIX"]["last"] == 17.8
    assert len(observation.history["SPY_close"]) == 30
    assert observation.provider_metadata["history_days"] == 30


def test_ensure_thread_event_loop_creates_one_when_missing(monkeypatch) -> None:
    policy = asyncio.get_event_loop_policy()
    original_get_event_loop = policy.get_event_loop

    def missing_loop():
        raise RuntimeError("There is no current event loop in thread 'ScriptRunner.scriptThread'.")

    monkeypatch.setattr(policy, "get_event_loop", missing_loop)
    try:
        loop = _ensure_thread_event_loop()
    finally:
        monkeypatch.setattr(policy, "get_event_loop", original_get_event_loop)

    assert isinstance(loop, asyncio.AbstractEventLoop)


def test_request_daily_history_surfaces_symbol_level_errors() -> None:
    class FakeIB:
        def reqHistoricalData(self, contract, **kwargs):
            raise RuntimeError(f"historical data denied for {kwargs['whatToShow']}")

    class FakeContract:
        secType = "IND"

    values, warnings = IBKRLiveClient._request_daily_history(
        FakeIB(),
        FakeContract(),
        history_days=30,
        history_label="VIX",
    )

    assert values == []
    assert warnings
    assert any("VIX history request failed for TRADES" in warning for warning in warnings)
    assert any("historical data denied" in warning for warning in warnings)


def test_validate_observation_preserves_provider_warnings() -> None:
    observation = ObservationRecord(
        schema_version="observation.v1",
        as_of="2026-06-10T20:00:00Z",
        source="IBKR",
        symbols={
            "SPY": {"last": 730.0},
            "VIX": {"last": None},
            "VVIX": {"last": None},
            "VIX9D": {"last": None},
            "VIX3M": {"last": None},
        },
        history={},
        quality={
            "warnings": ["VIX history request failed for TRADES: permission denied"],
            "stale_fields": ["VIX.last"],
        },
        option_chain={},
        provider_metadata={},
    )

    quality = validate_observation(observation)

    assert "VIX history request failed for TRADES: permission denied" in quality["warnings"]
