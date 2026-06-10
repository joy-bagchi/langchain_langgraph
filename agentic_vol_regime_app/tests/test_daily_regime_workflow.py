from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.contracts import MemoryQuery, MemoryRecord
from agentic_harness.stores import FilesystemMemoryStore
from agentic_vol_regime_app.contracts import AlertRecord, BeliefRecord, FeatureRecord, TransitionProbabilityRecord
from agentic_vol_regime_app.app_runtime import (
    default_ml_agent_path,
    default_hmm_agent_path,
    load_latest_live_daily_observation,
    load_recent_hmm_state_history,
    resume_daily_regime_run,
    run_daily_regime_agent,
)
import agentic_vol_regime_app.app_runtime as app_runtime
import agentic_vol_regime_app.pomdp.hmm_belief as hmm_belief
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


class RecordingDailyIBKRPipe:
    def __init__(self, as_of_sequence: list[str] | None = None) -> None:
        self.requests: list[int] = []
        self.as_of_sequence = list(as_of_sequence or [])
        self.call_count = 0

    def fetch_vol_regime_snapshot(self, request) -> object:
        current_index = self.call_count
        self.call_count += 1
        self.requests.append(int(request.history_days))
        if self.as_of_sequence:
            as_of = self.as_of_sequence[min(current_index, len(self.as_of_sequence) - 1)]
        else:
            as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        history_days = max(int(request.history_days), 0)
        history = {}
        if history_days > 0:
            history = {
                "SPY_close": [576.0 + (index * 0.9) for index in range(history_days)],
                "VIX": [16.1 + (index * 0.05) for index in range(history_days)],
                "VVIX": [91.0 + (index * 0.18) for index in range(history_days)],
                "VIX9D": [15.7 + (index * 0.04) for index in range(history_days)],
                "VIX3M": [19.7 + (index * 0.01) for index in range(history_days)],
            }

        class Snapshot:
            def to_dict(self_inner) -> dict:
                return {
                    "schema_version": "observation.v1",
                    "as_of": as_of,
                    "source": "IBKR",
                    "symbols": {
                        request.option_chain.symbol: {
                            "last": 602.25 + current_index,
                            "close": 601.84 + current_index,
                            "bid": 602.2 + current_index,
                            "ask": 602.3 + current_index,
                            "volume": 80421000,
                        },
                        "VIX": {"last": 17.4 + (current_index * 0.2)},
                        "VVIX": {"last": 96.1 + (current_index * 0.5)},
                        "VIX9D": {"last": 16.9 + (current_index * 0.1)},
                        "VIX3M": {"last": 19.6 + (current_index * 0.1)},
                    },
                    "history": history,
                    "quality": {"is_complete": True, "warnings": [], "stale_fields": []},
                    "option_chain": {
                        "underlying_symbol": request.option_chain.symbol,
                        "underlying_price": 602.25 + current_index,
                        "fetched_at": as_of,
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


class FakeGaussianHMM:
    def __init__(self, *, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
        self.means_ = None
        self.transmat_ = np.asarray(
            [
                [0.86, 0.09, 0.03, 0.02],
                [0.10, 0.72, 0.13, 0.05],
                [0.04, 0.18, 0.62, 0.16],
                [0.02, 0.09, 0.19, 0.70],
            ],
            dtype=float,
        )

    def fit(self, values: np.ndarray) -> "FakeGaussianHMM":
        self.means_ = np.repeat(np.linspace(-1.0, 1.0, 4).reshape(-1, 1), values.shape[1], axis=1)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([[0.58, 0.22, 0.14, 0.06] for _ in range(values.shape[0])], dtype=float)


def _load_sample_input(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "sample_inputs" / name).read_text(encoding="utf-8"))


def _run_and_complete_daily_agent(
    *,
    input_payload: dict,
    storage_root: Path,
    ibkr_data_pipe=None,
    agent_path: Path | None = None,
) -> dict:
    result = run_daily_regime_agent(
        input_payload=input_payload,
        storage_root=storage_root,
        ibkr_data_pipe=ibkr_data_pipe,
        agent_path=agent_path,
    )
    if result["status"] == "awaiting_review":
        result = resume_daily_regime_run(
            run_id=result["run_id"],
            decision="approved",
            notes="test approval",
            storage_root=storage_root,
            agent_path=agent_path,
        )
    return result


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


def test_daily_regime_ml_agent_completes_with_linear_model(tmp_path: Path) -> None:
    input_payload = _load_sample_input("daily_snapshot_watch.json")
    input_payload["report_root"] = str(tmp_path / "reports")

    result = run_daily_regime_agent(
        input_payload=input_payload,
        agent_path=default_ml_agent_path(),
        storage_root=tmp_path / ".workflow_memory",
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["belief_state"]["model_version"] == "linear_regression_regime_v1"
    assert result["agent"]["agent_id"] == "daily_regime_ml_orchestrator"
    assert result["named_outputs"]["daily_report"]["recommended_action"]


def test_daily_regime_hmm_agent_completes_with_hmm_advisory_output(tmp_path: Path) -> None:
    input_payload = _load_sample_input("daily_snapshot_watch.json")
    input_payload["report_root"] = str(tmp_path / "reports")

    original = hmm_belief.GaussianHMM
    hmm_belief.GaussianHMM = FakeGaussianHMM
    try:
        result = run_daily_regime_agent(
            input_payload=input_payload,
            agent_path=default_hmm_agent_path(),
            storage_root=tmp_path / ".workflow_memory",
        )
        initial_result = result

        if result["status"] == "awaiting_review":
            result = resume_daily_regime_run(
                run_id=result["run_id"],
                decision="approved",
                notes="reviewed",
                agent_path=default_hmm_agent_path(),
                storage_root=tmp_path / ".workflow_memory",
            )
    finally:
        hmm_belief.GaussianHMM = original

    assert result["status"] == "completed"
    assert initial_result["agent"]["agent_id"] == "daily_regime_hmm_orchestrator"
    assert "hmm_belief" in result["named_outputs"]
    assert "HMM Regime Persistence" in result["named_outputs"]["daily_report"]["markdown"]
    assert "Emission vs Persistence" in result["named_outputs"]["daily_report"]["markdown"]
    assert result["named_outputs"]["hmm_belief"]["interpretation_notes"]
    assert result["named_outputs"]["daily_report"]["comparison_panel"][2]["engine"] == "HMM"
    hmm_history = load_recent_hmm_state_history(
        agent_path=default_hmm_agent_path(),
        storage_root=tmp_path / ".workflow_memory",
    )
    if result["named_outputs"]["hmm_belief"]["is_trained"]:
        assert hmm_history
        assert hmm_history[0]["top_state"] == result["named_outputs"]["hmm_belief"]["top_state"]
    else:
        assert hmm_history == []


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


def test_daily_regime_ml_agent_supports_live_ibkr_input(tmp_path: Path) -> None:
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
        agent_path=default_ml_agent_path(),
        storage_root=tmp_path / ".workflow_memory",
        ibkr_data_pipe=FakeDailyIBKRPipe(),
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["belief_state"]["model_version"] == "linear_regression_regime_v1"
    latest_live_observation = load_latest_live_daily_observation(
        agent_path=default_ml_agent_path(),
        storage_root=tmp_path / ".workflow_memory",
    )
    assert latest_live_observation is not None
    assert latest_live_observation["symbols"]["VIX"]["last"] == 17.4


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


def test_daily_regime_workflow_reuses_cached_ibkr_history_within_24_hours(tmp_path: Path) -> None:
    pipe = RecordingDailyIBKRPipe()
    input_payload = {
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
    }

    storage_root = tmp_path / ".workflow_memory"
    first = _run_and_complete_daily_agent(
        input_payload=input_payload,
        storage_root=storage_root,
        ibkr_data_pipe=pipe,
        agent_path=default_hmm_agent_path(),
    )
    second = _run_and_complete_daily_agent(
        input_payload=input_payload,
        storage_root=storage_root,
        ibkr_data_pipe=pipe,
        agent_path=default_hmm_agent_path(),
    )

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert pipe.requests == [30, 0]
    assert len(second["named_outputs"]["observation"]["history"]["SPY_close"]) == 30
    assert second["named_outputs"]["observation"]["provider_metadata"]["history_cache_mode"] == "cache_reuse"
    assert second["named_outputs"]["observation"]["provider_metadata"]["history_requested_from_ibkr"] == 0


def test_daily_regime_workflow_refreshes_cached_ibkr_history_after_24_hours(tmp_path: Path) -> None:
    storage_root = tmp_path / ".workflow_memory"
    namespace = "daily_regime_memory"
    current_as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    pipe = RecordingDailyIBKRPipe(as_of_sequence=[current_as_of])
    memory_store = FilesystemMemoryStore(storage_root)
    services = build_platform_services(
        storage_root=storage_root,
        memory_service_type="semantic",
    )
    old_as_of = (datetime.now(timezone.utc) - timedelta(days=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    services.memory.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="regime_history_cache",
            content="latest IBKR regime history cache",
            source_run_id="seed-run",
            source_step_id="seed-step",
            metadata={
                "source_kind": "ibkr_regime_history_cache",
                "observation_as_of": old_as_of,
            },
            structured_payload={
                "source_kind": "ibkr_regime_history_cache",
                "observation_as_of": old_as_of,
                "last_history_refresh_at": old_as_of,
                "history_days": 30,
                "history_refresh_mode": "full_fetch",
                "history": {
                    "SPY_close": [576.0 + (index * 0.9) for index in range(30)],
                    "VIX": [16.1 + (index * 0.05) for index in range(30)],
                    "VVIX": [91.0 + (index * 0.18) for index in range(30)],
                    "VIX9D": [15.7 + (index * 0.04) for index in range(30)],
                    "VIX3M": [19.7 + (index * 0.01) for index in range(30)],
                },
            },
        )
    )

    result = _run_and_complete_daily_agent(
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
        storage_root=storage_root,
        ibkr_data_pipe=pipe,
        agent_path=default_hmm_agent_path(),
    )

    cache_matches = memory_store.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=1,
            memory_types=["regime_history_cache"],
            structured_filters={"source_kind": "ibkr_regime_history_cache"},
        )
    )

    assert result["status"] == "completed"
    assert pipe.requests == [1]
    assert len(result["named_outputs"]["observation"]["history"]["SPY_close"]) == 30
    assert result["named_outputs"]["observation"]["provider_metadata"]["history_cache_mode"] == "incremental_refresh"
    assert result["named_outputs"]["observation"]["provider_metadata"]["history_requested_from_ibkr"] == 1
    assert cache_matches
    assert cache_matches[0].record.structured_payload["last_history_refresh_at"] == current_as_of
    assert len(cache_matches[0].record.structured_payload["history"]["SPY_close"]) == 30


def test_daily_regime_heuristic_agent_bypasses_hmm_history_cache(tmp_path: Path) -> None:
    pipe = RecordingDailyIBKRPipe()
    input_payload = {
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
    }
    storage_root = tmp_path / ".workflow_memory"

    first = _run_and_complete_daily_agent(
        input_payload=input_payload,
        storage_root=storage_root,
        ibkr_data_pipe=pipe,
    )
    second = _run_and_complete_daily_agent(
        input_payload=input_payload,
        storage_root=storage_root,
        ibkr_data_pipe=pipe,
    )

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert pipe.requests == [30, 30]
    assert second["named_outputs"]["observation"]["provider_metadata"]["history_cache_mode"] == "disabled_for_engine"


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


def test_app_runtime_emits_agentic_vol_regime_app_trace_layer(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, list[dict]] = {"trace_calls": [], "trace_outputs": []}

    class FakeClient:
        def flush(self, timeout=None) -> None:
            return None

    @contextmanager
    def fake_tracing_context(**kwargs):
        yield

    class FakeTrace:
        def __init__(self, **kwargs) -> None:
            captured["trace_calls"].append(kwargs)
            self.metadata = dict(kwargs.get("metadata") or {})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def end(self, *, outputs=None):
            captured["trace_outputs"].append(outputs)

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="semantic",
        langsmith_tracing=True,
        langsmith_project="agentic-harness-tests",
        langsmith_api_key="test-key",
        langsmith_client=FakeClient(),
        ibkr_data_pipe=FakeDailyIBKRPipe(),
    )
    services.observability._tracing_context_factory = fake_tracing_context
    services.observability._trace_factory = lambda **kwargs: FakeTrace(**kwargs)

    monkeypatch.setattr(app_runtime, "build_platform_services", lambda **kwargs: services)

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
        langsmith_tracing=True,
    )

    span_names = [call["name"] for call in captured["trace_calls"]]

    assert result["status"] == "completed"
    assert "agentic_vol_regime_app:run_daily_regime_agent" in span_names
