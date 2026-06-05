from __future__ import annotations

import json
from pathlib import Path

from agentic_vol_regime_app.contracts import AlertRecord, BeliefRecord, FeatureRecord, TransitionProbabilityRecord
from agentic_vol_regime_app.app_runtime import (
    load_latest_live_daily_observation,
    resume_daily_regime_run,
    run_daily_regime_agent,
)
from agentic_vol_regime_app.pomdp.policy import recommend_policy_action


class FakeDailyIBKRPipe:
    def fetch_vol_regime_snapshot(self, request) -> object:
        class Snapshot:
            def to_dict(self_inner) -> dict:
                return {
                    "schema_version": "observation.v1",
                    "as_of": "2026-06-04T20:00:00Z",
                    "source": "IBKR",
                    "symbols": {
                        request.option_chain.symbol: {
                            "last": 602.25,
                            "close": 601.84,
                            "bid": 602.2,
                            "ask": 602.3,
                            "volume": 80421000,
                        },
                        "VIX": {"last": 17.4},
                        "VVIX": {"last": 96.1},
                        "VIX9D": {"last": 16.9},
                        "VIX3M": {"last": 19.6},
                    },
                    "history": {
                        "SPY_close": [576.0 + (index * 0.9) for index in range(30)],
                        "VIX": [16.1 + (index * 0.05) for index in range(30)],
                        "VVIX": [91.0 + (index * 0.18) for index in range(30)],
                        "VIX9D": [15.7 + (index * 0.04) for index in range(30)],
                        "VIX3M": [19.7 + (index * 0.01) for index in range(30)],
                    },
                    "quality": {"is_complete": True, "warnings": [], "stale_fields": []},
                    "option_chain": {
                        "underlying_symbol": request.option_chain.symbol,
                        "underlying_price": 602.25,
                        "fetched_at": "2026-06-04T20:00:00Z",
                        "exchange": request.option_chain.option_exchange,
                        "currency": request.option_chain.currency,
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
                    "provider_metadata": {"port": 4001, "history_days": request.history_days},
                }

        return Snapshot()


class FakeDailyIBKRSentinelPipe:
    def fetch_vol_regime_snapshot(self, request) -> object:
        class Snapshot:
            def to_dict(self_inner) -> dict:
                return {
                    "schema_version": "observation.v1",
                    "as_of": "2026-06-04T20:00:00Z",
                    "source": "IBKR",
                    "symbols": {
                        request.option_chain.symbol: {
                            "last": 602.25,
                            "close": 601.84,
                            "bid": 602.2,
                            "ask": 602.3,
                            "volume": 80421000,
                        },
                        "VIX": {"last": None},
                        "VVIX": {"last": None},
                        "VIX9D": {"last": 16.9},
                        "VIX3M": {"last": 19.6},
                    },
                    "history": {
                        "SPY_close": [576.0 + (index * 0.9) for index in range(30)],
                        "VIX9D": [15.7 + (index * 0.04) for index in range(30)],
                        "VIX3M": [19.7 + (index * 0.01) for index in range(30)],
                    },
                    "quality": {
                        "is_complete": False,
                        "warnings": ["VIX and VVIX unavailable from live quote"],
                        "stale_fields": ["VIX.last", "VVIX.last"],
                        "missing_history": ["VIX", "VVIX"],
                    },
                    "option_chain": {
                        "underlying_symbol": request.option_chain.symbol,
                        "underlying_price": 602.25,
                        "fetched_at": "2026-06-04T20:00:00Z",
                        "exchange": request.option_chain.option_exchange,
                        "currency": request.option_chain.currency,
                        "expirations": ["20260620"],
                        "strikes": [600.0],
                        "rights": ["C", "P"],
                        "option_quotes": [],
                    },
                    "provider_metadata": {"port": 4001, "history_days": request.history_days},
                }

        return Snapshot()


def _load_sample_input(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "sample_inputs" / name).read_text(encoding="utf-8"))


def test_daily_regime_workflow_completes_and_writes_report(tmp_path: Path) -> None:
    input_payload = _load_sample_input("daily_snapshot_watch.json")
    input_payload["report_root"] = str(tmp_path / "reports")

    result = run_daily_regime_agent(
        input_payload=input_payload,
        storage_root=tmp_path / ".workflow_memory",
    )

    assert result["status"] == "completed"
    daily_report = result["named_outputs"]["daily_report"]
    assert "Daily Volatility Regime Report" in daily_report["markdown"]
    assert Path(daily_report["report_path"]).exists()
    assert result["named_outputs"]["belief_state"]["beliefs"]["STABLE_LOW_VOL_TREND"] > 0.0
    assert result["named_outputs"]["memory_candidates"]["candidate_count"] >= 0


def test_high_risk_run_pauses_for_human_review_and_resumes(tmp_path: Path) -> None:
    input_payload = {
        "market_snapshot": {
            "schema_version": "observation.v1",
            "as_of": "2026-05-30T20:00:00Z",
            "source": "critical_fixture",
            "symbols": {
                "SPY": {"last": 543.2, "close": 543.2, "volume": 118000000},
                "VIX": {"last": 34.8},
                "VVIX": {"last": 158.0},
                "VIX9D": {"last": 37.1},
                "VIX3M": {"last": 31.6}
            },
            "history": {
                "SPY_close": [598.0, 596.2, 593.8, 591.1, 588.0, 585.4, 582.3, 579.1, 576.4, 573.0, 570.5, 568.2, 566.0, 563.8, 561.1, 558.5, 556.1, 553.9, 551.6, 549.8, 548.1, 546.7, 545.8, 544.4, 543.2],
                "VIX": [20.0, 20.5, 20.2, 21.0, 21.4, 22.0, 22.7, 23.3, 23.9, 24.6, 25.1, 25.8, 26.4, 27.1, 27.9, 28.8, 29.7, 30.6, 31.2, 31.8, 32.6, 33.4, 34.1, 34.5, 34.8],
                "VVIX": [102.0, 103.5, 104.0, 106.2, 107.5, 109.0, 111.2, 113.0, 115.0, 118.0, 120.5, 123.0, 126.2, 129.5, 132.8, 136.0, 139.5, 143.2, 146.0, 149.5, 152.0, 154.2, 156.0, 157.0, 158.0],
                "VIX9D": [18.5, 19.0, 19.2, 19.8, 20.4, 20.9, 21.6, 22.5, 23.2, 24.0, 24.8, 25.6, 26.5, 27.5, 28.7, 29.8, 31.0, 32.1, 33.0, 34.0, 35.0, 35.8, 36.3, 36.8, 37.1],
                "VIX3M": [26.5, 26.2, 26.0, 25.8, 25.7, 25.5, 25.3, 25.2, 25.0, 24.9, 24.7, 24.5, 24.4, 24.2, 24.0, 23.9, 23.8, 23.6, 23.5, 23.3, 23.2, 23.1, 23.0, 22.9, 31.6]
            }
        },
        "report_root": str(tmp_path / "reports")
    }

    result = run_daily_regime_agent(
        input_payload=input_payload,
        storage_root=tmp_path / ".workflow_memory",
    )

    assert result["status"] == "awaiting_review"
    assert result["pending_review"]["step_id"] == "human_review_gate"

    resumed = resume_daily_regime_run(
        run_id=result["run_id"],
        decision="approved",
        notes="reviewed",
        storage_root=tmp_path / ".workflow_memory",
    )

    assert resumed["status"] == "completed"
    assert resumed["named_outputs"]["review_decision"]["decision"] == "approved"
    assert Path(resumed["named_outputs"]["daily_report"]["report_path"]).exists()


def test_daily_regime_workflow_supports_live_ibkr_input(tmp_path: Path) -> None:
    result = run_daily_regime_agent(
        input_payload={
            "data_provider": "ibkr",
            "symbol": "SPY",
            "ibkr": {
                "host": "127.0.0.1",
                "port": 4001,
                "client_id": 73,
                "market_data_type": 1,
                "exchange": "SMART",
                "option_exchange": "SMART",
                "currency": "USD",
                "index_exchange": "CBOE",
                "expiry_count": 1,
                "strike_count": 1,
                "history_days": 30,
            },
            "report_root": str(tmp_path / "reports"),
        },
        storage_root=tmp_path / ".workflow_memory",
        ibkr_data_pipe=FakeDailyIBKRPipe(),
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["observation"]["source"] == "IBKR"
    assert result["named_outputs"]["observation"]["provider_metadata"]["port"] == 4001
    assert result["named_outputs"]["observation"]["option_chain"]["option_quotes"][0]["greeks"]["delta"] == 0.51
    assert result["named_outputs"]["daily_report"]["recommended_action"] == "NO_OVERWRITE"
    latest_live_observation = load_latest_live_daily_observation(
        storage_root=tmp_path / ".workflow_memory",
    )
    assert latest_live_observation is not None
    assert latest_live_observation["symbols"]["SPY"]["last"] == 602.25


def test_daily_regime_workflow_backfills_missing_live_vol_quotes(tmp_path: Path) -> None:
    sample_input = _load_sample_input("daily_snapshot_watch.json")
    result = run_daily_regime_agent(
        input_payload={
            "data_provider": "ibkr",
            "symbol": "SPY",
            "ibkr": {
                "host": "127.0.0.1",
                "port": 4001,
                "client_id": 73,
                "market_data_type": 1,
                "exchange": "SMART",
                "option_exchange": "SMART",
                "currency": "USD",
                "index_exchange": "CBOE",
                "expiry_count": 1,
                "strike_count": 1,
                "history_days": 30,
            },
            "reference_market_snapshot": dict(sample_input["market_snapshot"]),
            "report_root": str(tmp_path / "reports"),
        },
        storage_root=tmp_path / ".workflow_memory",
        ibkr_data_pipe=FakeDailyIBKRSentinelPipe(),
    )

    assert result["status"] == "completed"
    observation = result["named_outputs"]["observation"]
    assert observation["symbols"]["VIX"]["last"] == 18.4
    assert observation["symbols"]["VVIX"]["last"] == 101.0
    assert observation["history"]["VIX"][-1] == 18.4
    assert observation["quality"]["reference_backfill_applied"] is True


def test_policy_recommendation_emits_overwrite_strike_and_dte() -> None:
    feature_record = FeatureRecord(
        schema_version="features.v1",
        as_of="2026-06-04T20:00:00Z",
        feature_set_version="vol_regime_features_v1",
        features={"spy_last": 757.0},
        missing_features=[],
        lookback_windows={},
    )
    belief_record = BeliefRecord(
        schema_version="belief.v1",
        as_of="2026-06-04T20:00:00Z",
        model_version="heuristic",
        beliefs={
            "STABLE_LOW_VOL_TREND": 0.18,
            "VOL_EXPANSION_TRANSITION": 0.41,
            "HIGH_VOL_RISK_OFF": 0.17,
            "PANIC_CONVEXITY_STRESS": 0.05,
            "POST_PANIC_COMPRESSION": 0.05,
        },
        belief_delta={},
        entropy=0.8,
        confidence=0.64,
        drivers=[],
    )
    transition_record = TransitionProbabilityRecord(
        schema_version="transition.v1",
        as_of="2026-06-04T20:00:00Z",
        model_version="heuristic",
        transition_probabilities={"vol_expansion_5d": 0.36, "risk_off_transition_10d": 0.12},
        top_predictive_factors=[],
        confirming_features_count=3,
    )
    alert_record = AlertRecord(
        schema_version="alert.v1",
        alert_id="alert-1",
        as_of="2026-06-04T20:00:00Z",
        severity="WARNING",
        alert_type="vol_transition",
        headline="Expansion risk is building.",
        probabilities={},
        belief_state={},
        drivers=[],
        recommended_review=[],
        requires_human_review=False,
    )

    recommendation = recommend_policy_action(
        feature_record,
        belief_record,
        transition_record,
        alert_record,
    )

    assert recommendation.recommended_action == "MEDIUM_OVERWRITE"
    assert recommendation.overwrite_call_strike == 764.0
    assert recommendation.overwrite_dte == 1
