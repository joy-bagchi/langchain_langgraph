from __future__ import annotations

import json
from pathlib import Path

from agentic_vol_regime_app.app_runtime import resume_daily_regime_run, run_daily_regime_agent


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
