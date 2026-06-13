from __future__ import annotations

import pandas as pd

from src.backtest.hmm_replay.replay_report import build_replay_report_markdown
from src.backtest.hmm_replay.replay_scoring import (
    build_disagreement_attribution,
    build_disagreement_summary,
    build_geometry_case_files,
    build_geometry_false_suppression_analysis,
    score_prediction,
)


def _prediction(as_of: str, model: str, state: str) -> dict:
    return {
        "as_of_date": as_of,
        "model_name": model,
        "top_state": state,
        "state_probabilities": {
            "STABLE_LOW_VOL_TREND": 0.25,
            "MID_VOL_CHOP": 0.25,
            "VOL_EXPANSION_TRANSITION": 0.25,
            "HIGH_VOL_RISK_OFF": 0.25,
        },
        "transition_probabilities": {"to_higher_vol_1d": 0.4, "to_higher_vol_2d": 0.5, "to_higher_vol_3d": 0.6},
        "policy_output": {"overwrite_posture": "LIGHT_OVERWRITE"},
        "feature_snapshot": {
            "vix": 16.0,
            "vvix": 88.0,
            "vvix_vix_ratio": 5.5,
            "vvix_vix_z_22d": 0.2,
            "vix_vix3m_ratio": 0.9,
            "vix9d_vix_ratio": 0.95,
            "term_structure_slope": 1.2,
            "realized_vol_5d": 11.0,
            "realized_vol_21d": 13.0,
            "spy_return_1d": 0.001,
            "drawdown_21d": 0.03,
            "trend_persistence_21d": 0.6,
            "avg_pairwise_corr_21d": 0.5,
            "first_eigenvalue_share_21d": 0.4,
            "effective_rank_21d": 4.1,
            "log_det_corr_21d": -2.2,
        },
    }


def _outcome(as_of: str, r1: str, r2: str, r3: str, v1: float, v2: float, v3: float) -> dict:
    return {
        "as_of_date": as_of,
        "rv21_asof": 14.0,
        "realized_regime_label_1d": r1,
        "realized_regime_label_2d": r2,
        "realized_regime_label_3d": r3,
        "vix_change_1d": v1,
        "vix_change_2d": v2,
        "vix_change_3d": v3,
        "vvix_change_1d": v1 * 1.1,
        "vvix_change_2d": v2 * 1.1,
        "vvix_change_3d": v3 * 1.1,
        "rv21_change_1d": 0.2 if r1 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"} else -0.2,
        "rv21_change_2d": 0.2 if r2 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"} else -0.2,
        "rv21_change_3d": 0.2 if r3 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"} else -0.2,
        "spy_return_1d": 0.001,
        "spy_return_2d": 0.001,
        "spy_return_3d": 0.001,
        "vix_rose_1d": v1 > 0.0,
        "vix_rose_2d": v2 > 0.0,
        "vix_rose_3d": v3 > 0.0,
        "rv_expanded_1d": r1 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"},
        "rv_expanded_2d": r2 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"},
        "rv_expanded_3d": r3 in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"},
        "vix_spike_1d": v1 >= 0.10,
        "vix_spike_2d": v2 >= 0.10,
        "vix_spike_3d": v3 >= 0.10,
    }


def test_disagreement_attribution_slice2() -> None:
    horizons = [1, 2, 3]
    predictions = [
        _prediction("2026-06-10", "hmm_v3_core_plus_sector_geometry", "MID_VOL_CHOP"),
        _prediction("2026-06-10", "hmm_v1_core", "VOL_EXPANSION_TRANSITION"),
        _prediction("2026-06-10", "hmm_v2_core_plus_sector_corr", "VOL_EXPANSION_TRANSITION"),
        _prediction("2026-06-10", "heuristic", "VOL_EXPANSION_TRANSITION"),
        _prediction("2026-06-11", "hmm_v3_core_plus_sector_geometry", "VOL_EXPANSION_TRANSITION"),
        _prediction("2026-06-11", "hmm_v1_core", "MID_VOL_CHOP"),
        _prediction("2026-06-11", "hmm_v2_core_plus_sector_corr", "MID_VOL_CHOP"),
        _prediction("2026-06-11", "heuristic", "MID_VOL_CHOP"),
    ]
    outcomes = [
        _outcome("2026-06-10", "MID_VOL_CHOP", "MID_VOL_CHOP", "MID_VOL_CHOP", -0.02, -0.015, -0.01),
        _outcome("2026-06-11", "VOL_EXPANSION_TRANSITION", "VOL_EXPANSION_TRANSITION", "MID_VOL_CHOP", 0.03, 0.02, 0.0),
    ]
    scored_records: list[dict] = []
    for prediction in predictions:
        outcome = next(item for item in outcomes if item["as_of_date"] == prediction["as_of_date"])
        for horizon in horizons:
            scored_records.append(score_prediction(prediction, outcome, horizon=horizon))

    disagreement = build_disagreement_attribution(
        predictions=predictions,
        outcomes=outcomes,
        scored_records=scored_records,
        horizons=horizons,
    )
    assert not disagreement.empty
    assert (disagreement["comparison_model"] == "hmm_v1_core").any()
    assert (disagreement["disagreement_type"] == "v3_downgrade").any()
    assert (disagreement["disagreement_type"] == "v3_upgrade").any()
    assert (disagreement["is_opposite_bucket"] == True).any()  # noqa: E712

    row = disagreement[(disagreement["as_of_date"] == "2026-06-10") & (disagreement["comparison_model"] == "hmm_v1_core")].iloc[0]
    assert row["v3_won_1d"] in {"true", "false", "tie"}
    assert row["v3_bucket_won_1d"] in {"true", "false", "tie"}

    summary = build_disagreement_summary(disagreement, horizons=horizons)
    assert not summary.empty
    assert "v3_win_rate" in summary.columns
    assert "false_suppression_rate" in summary.columns
    assert "false_alarm_filter_success_rate" in summary.columns

    override, false_suppression, success = build_geometry_case_files(disagreement, horizons=horizons)
    assert not override.empty
    assert not success.empty
    assert "win_horizon" in success.columns
    assert isinstance(false_suppression, pd.DataFrame)

    report = build_replay_report_markdown(
        summary_metrics=pd.DataFrame(),
        recent_rows=pd.DataFrame(),
        disagreements=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
        prediction_distribution=pd.DataFrame(),
        outcome_distribution=pd.DataFrame(),
        confusion_matrix=pd.DataFrame(),
        economic_summary=pd.DataFrame(),
        false_alarms=pd.DataFrame(),
        missed_risks=pd.DataFrame(),
        disagreement_attribution=disagreement,
        disagreement_summary=summary,
        geometry_override_cases=override,
        geometry_false_suppression_cases=false_suppression,
        geometry_false_suppression_analysis=build_geometry_false_suppression_analysis(
            false_suppression,
            horizons=horizons,
        ),
        geometry_success_cases=success,
        geometry_smooth_modifier=pd.DataFrame(),
    )
    assert "## Disagreement Attribution" in report


def test_geometry_false_suppression_analysis_includes_required_metrics() -> None:
    sample = pd.DataFrame(
        [
            {
                "as_of_date": "2026-06-10",
                "v3_result": "win",
                "vix": 20.0,
                "vvix": 100.0,
                "vvix_vix_ratio": 5.0,
                "geometry_stress_score": 0.25,
                "downgrade_levels": 1,
                "realized_state_1d": "VOL_EXPANSION_TRANSITION",
                "realized_risk_bucket_1d": "HIGHER_VOL_RISK",
            },
            {
                "as_of_date": "2026-06-11",
                "v3_result": "loss",
                "vix": 16.0,
                "vvix": 90.0,
                "vvix_vix_ratio": 5.625,
                "geometry_stress_score": 0.15,
                "downgrade_levels": 2,
                "realized_state_1d": "MID_VOL_CHOP",
                "realized_risk_bucket_1d": "LOW_RISK",
            },
        ]
    )
    analysis = build_geometry_false_suppression_analysis(sample, horizons=[1])

    assert not analysis.empty
    assert (analysis["metric"] == "count_by_realized_outcome").any()
    assert (analysis["metric"] == "count_by_realized_risk_bucket").any()
    assert (analysis["metric"] == "avg_vix").any()
    assert (analysis["metric"] == "avg_vvix").any()
    assert (analysis["metric"] == "avg_vvix_vix_ratio").any()
    assert (analysis["metric"] == "avg_geometry_stress_score").any()
    assert (analysis["metric"] == "avg_downgrade_levels").any()
    assert set(analysis["segment"].unique()) >= {"overall", "wins", "losses"}
