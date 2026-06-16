from __future__ import annotations

import pandas as pd

from src.backtest.hmm_replay.replay_report import build_replay_report_markdown
from src.backtest.hmm_replay.replay_scoring import (
    build_false_alarm_and_missed_risk_reports,
    normalize_regime_label,
    regime_risk_bucket,
    regime_severity,
    score_prediction,
)


def _base_prediction(state: str, *, model_name: str = "hmm_v3_core_plus_sector_geometry") -> dict:
    return {
        "as_of_date": "2026-06-10",
        "model_name": model_name,
        "top_state": state,
        "state_probabilities": {
            "STABLE_LOW_VOL_TREND": 0.25,
            "MID_VOL_CHOP": 0.25,
            "VOL_EXPANSION_TRANSITION": 0.25,
            "HIGH_VOL_RISK_OFF": 0.25,
        },
        "transition_probabilities": {
            "to_higher_vol_1d": 0.8,
            "to_higher_vol_2d": 0.8,
            "to_higher_vol_3d": 0.8,
        },
        "policy_output": {"overwrite_posture": "MEDIUM_OVERWRITE"},
        "feature_snapshot": {
            "vix": 16.0,
            "vvix_vix_ratio": 5.4,
            "term_structure_slope": 1.2,
            "avg_pairwise_corr_21d": 0.5,
            "first_eigenvalue_share_21d": 0.44,
            "effective_rank_21d": 4.2,
            "log_det_corr_21d": -2.0,
        },
    }


def _base_outcome(*, realized: str, vix_change: float, rv21_change: float, vvix_change: float = 0.02) -> dict:
    return {
        "as_of_date": "2026-06-10",
        "rv21_asof": 14.0,
        "realized_regime_label_1d": realized,
        "vix_change_1d": vix_change,
        "vvix_change_1d": vvix_change,
        "rv21_change_1d": rv21_change,
        "spy_return_1d": 0.002,
        "vix_rose_1d": bool(vix_change > 0.0),
        "rv_expanded_1d": bool(rv21_change > 0.0),
        "vix_spike_1d": bool(vix_change >= 0.10),
    }


def test_regime_mapping_and_risk_bucket() -> None:
    assert normalize_regime_label("stable_low_vol") == "STABLE_LOW_VOL_TREND"
    assert regime_severity("STABLE_LOW_VOL_TREND") == 0
    assert regime_severity("HIGH_VOL_RISK_OFF") == 3
    assert regime_risk_bucket("MID_VOL_CHOP") == "LOW_RISK"
    assert regime_risk_bucket("VOL_EXPANSION_TRANSITION") == "HIGHER_VOL_RISK"


def test_adjacent_and_severe_miss_logic() -> None:
    adjacent = score_prediction(
        _base_prediction("STABLE_LOW_VOL_TREND"),
        _base_outcome(realized="MID_VOL_CHOP", vix_change=-0.02, rv21_change=-0.1),
        horizon=1,
    )
    severe = score_prediction(
        _base_prediction("STABLE_LOW_VOL_TREND"),
        _base_outcome(realized="HIGH_VOL_RISK_OFF", vix_change=0.12, rv21_change=0.9),
        horizon=1,
    )
    assert adjacent["adjacent_correct"] is True
    assert adjacent["severe_miss"] is False
    assert severe["adjacent_correct"] is False
    assert severe["severe_miss"] is True


def test_brier_metric_unchanged_definition() -> None:
    scored = score_prediction(
        _base_prediction("VOL_EXPANSION_TRANSITION"),
        _base_outcome(realized="VOL_EXPANSION_TRANSITION", vix_change=0.02, rv21_change=0.3),
        horizon=1,
    )
    assert abs(scored["brier_vol_expansion"] - 0.04) < 1e-9
    assert abs(scored["brier_vix_spike"] - 0.64) < 1e-9


def test_directional_vix_accuracy_and_risk_flags() -> None:
    scored = score_prediction(
        _base_prediction("VOL_EXPANSION_TRANSITION"),
        _base_outcome(realized="VOL_EXPANSION_TRANSITION", vix_change=0.03, rv21_change=0.2),
        horizon=1,
    )
    assert scored["directional_vix_correct"] is True
    assert scored["risk_bucket_correct"] is True
    assert scored["false_alarm"] is False
    assert scored["missed_risk"] is False


def test_false_alarm_and_missed_risk_reports() -> None:
    false_alarm_scored = score_prediction(
        _base_prediction("VOL_EXPANSION_TRANSITION"),
        _base_outcome(realized="MID_VOL_CHOP", vix_change=-0.01, rv21_change=-0.2),
        horizon=1,
    )
    missed_risk_scored = score_prediction(
        _base_prediction("MID_VOL_CHOP"),
        _base_outcome(realized="HIGH_VOL_RISK_OFF", vix_change=0.11, rv21_change=0.7),
        horizon=1,
    )
    false_alarms, missed_risks = build_false_alarm_and_missed_risk_reports(
        scored_records=[false_alarm_scored, missed_risk_scored],
        predictions=[
            _base_prediction("VOL_EXPANSION_TRANSITION"),
            _base_prediction("MID_VOL_CHOP", model_name="hmm_v3_core_plus_sector_geometry"),
        ],
        outcomes=[
            _base_outcome(realized="MID_VOL_CHOP", vix_change=-0.01, rv21_change=-0.2),
        ],
    )
    assert not false_alarms.empty
    assert not missed_risks.empty


def test_report_renders_new_sections() -> None:
    markdown = build_replay_report_markdown(
        summary_metrics=pd.DataFrame([{"model_name": "hmm_v3_core_plus_sector_geometry", "horizon": 1, "accuracy": 0.5}]),
        recent_rows=pd.DataFrame(),
        disagreements=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
        prediction_distribution=pd.DataFrame(),
        outcome_distribution=pd.DataFrame(),
        confusion_matrix=pd.DataFrame(),
        economic_summary=pd.DataFrame(
            [
                {
                    "model_name": "hmm_v3_core_plus_sector_geometry",
                    "horizon": 1,
                    "accuracy": 0.42,
                    "adjacent_tolerant_accuracy": 0.73,
                    "missed_risk_rate": 0.12,
                    "false_alarm_rate": 0.10,
                    "brier_vix_spike": 0.09,
                    "combined_vol_directional_accuracy": 0.62,
                }
            ]
        ),
        false_alarms=pd.DataFrame(),
        missed_risks=pd.DataFrame(),
        disagreement_attribution=pd.DataFrame(),
        disagreement_summary=pd.DataFrame(),
        geometry_override_cases=pd.DataFrame(),
        geometry_false_suppression_cases=pd.DataFrame(),
        geometry_false_suppression_analysis=pd.DataFrame(),
        geometry_success_cases=pd.DataFrame(),
        geometry_smooth_modifier=pd.DataFrame(),
        path_aware_meta_learner=pd.DataFrame(),
        path_feature_diagnostics=pd.DataFrame(),
    )
    assert "## Prediction Distribution" in markdown
    assert "## Outcome Distribution" in markdown
    assert "## Confusion Matrix by Horizon" in markdown
    assert "## Economic Score Summary" in markdown
    assert "## Model Usefulness Summary" in markdown
    assert "## Geometry Smooth Modifier" in markdown
    assert "## Path-Aware Meta Learner" in markdown
    assert "## Path Feature Diagnostics" in markdown
