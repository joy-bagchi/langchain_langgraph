from __future__ import annotations

from pathlib import Path

from agentic_vol_regime_app.app_runtime import run_live_overwrite_policy_engine
from agentic_vol_regime_app.contracts import ObservationRecord


class FakeIBKRDataPipe:
    def __init__(self) -> None:
        self.order_call_count = 0

    def fetch_vol_regime_snapshot(self, _request) -> ObservationRecord:
        return ObservationRecord(
            schema_version="observation.v1",
            as_of="2026-06-25T20:00:00Z",
            source="IBKR",
            symbols={
                "SPY": {"last": 740.25, "close": 739.90},
                "VIX": {"last": 16.8},
            },
            history={},
            quality={"is_complete": True, "warnings": [], "stale_fields": []},
            option_chain={
                "option_quotes": [
                    {
                        "symbol": "SPY_743C",
                        "expiry": "20260626",
                        "strike": 743.0,
                        "right": "C",
                        "bid": 1.85,
                        "ask": 1.95,
                        "mark": 1.90,
                        "greeks": {"delta": 0.22, "implied_vol": 0.16},
                    },
                    {
                        "symbol": "SPY_745C",
                        "expiry": "20260626",
                        "strike": 745.0,
                        "right": "C",
                        "bid": 1.90,
                        "ask": 2.00,
                        "mark": 1.95,
                        "greeks": {"delta": 0.20, "implied_vol": 0.16},
                    },
                    {
                        "symbol": "SPY_746C",
                        "expiry": "20260627",
                        "strike": 746.0,
                        "right": "C",
                        "bid": 1.20,
                        "ask": 1.35,
                        "mark": 1.275,
                        "greeks": {"delta": 0.18, "implied_vol": 0.16},
                    },
                ]
            },
            provider_metadata={"host": "127.0.0.1", "port": 4001, "client_id": 73},
        )

    def place_order(self, *args, **kwargs) -> None:  # pragma: no cover
        self.order_call_count += 1
        raise AssertionError("Overwrite policy engine must not place orders.")


def test_live_overwrite_policy_engine_with_mocked_ibkr(monkeypatch, tmp_path: Path) -> None:
    fake_pipe = FakeIBKRDataPipe()

    def fake_run_daily_regime_agent(**kwargs):
        assert kwargs["agent_path"]
        return {
            "named_outputs": {
                "hmm_belief": {
                    "as_of": "2026-06-25T20:00:00Z",
                    "top_state": "VOL_EXPANSION",
                    "state_probabilities": {
                        "LOW_VOL_TREND": 0.10,
                        "MID_VOL_CHOP": 0.20,
                        "VOL_EXPANSION": 0.60,
                        "HIGH_VOL_RISK_OFF": 0.10,
                    },
                }
            }
        }

    monkeypatch.setattr("agentic_vol_regime_app.app_runtime.run_daily_regime_agent", fake_run_daily_regime_agent)

    result = run_live_overwrite_policy_engine(
        underlying="SPY",
        regime_engine="HMMv1 Agent",
        leap_contracts=5,
        leap_delta=0.80,
        base_min_premium=1.40,
        dte_choices=[1],
        strikes_below_target=2,
        strikes_above_target=2,
        host="127.0.0.1",
        port=4001,
        client_id=73,
        market_data_type=1,
        output_dir=str(tmp_path / "overwrite_live"),
        ibkr_data_pipe=fake_pipe,
        langsmith_tracing=False,
    )

    assert result["recommendation_mode"] == "SELECTIVE_ONLY"
    assert result["top_accepted_candidates"]
    assert Path(result["scored_candidates_path"]).exists()
    assert Path(result["scenario_pnl_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert Path(result["live_snapshot_path"]).exists()
    assert fake_pipe.order_call_count == 0
