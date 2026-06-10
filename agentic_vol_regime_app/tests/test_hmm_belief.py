from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.contracts import (
    AlertRecord,
    BeliefRecord,
    CriticReviewRecord,
    FeatureRecord,
    HMMBeliefRecord,
    ObservationRecord,
    PolicyRecommendationRecord,
    TransitionProbabilityRecord,
)
import agentic_vol_regime_app.pomdp.hmm_belief as hmm_belief
from agentic_vol_regime_app.pomdp.hmm_belief import (
    HMMConfig,
    _build_historical_feature_rows,
    _matrix_power,
    _repair_transition_matrix,
    _safe_expected_duration,
    compute_hmm_belief_record,
    hmm_to_belief_record,
)
from agentic_vol_regime_app.reports.daily_report import render_daily_markdown


class FakeGaussianHMM:
    last_fit_rows = 0
    fit_calls = 0

    def __init__(self, *, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.means_ = None
        if self.n_components == 3:
            self.transmat_ = np.asarray(
                [
                    [0.84, 0.12, 0.04],
                    [0.10, 0.72, 0.18],
                    [0.04, 0.16, 0.80],
                ],
                dtype=float,
            )
        else:
            self.transmat_ = np.asarray(
                [
                    [0.82, 0.12, 0.04, 0.02],
                    [0.10, 0.70, 0.15, 0.05],
                    [0.04, 0.18, 0.60, 0.18],
                    [0.02, 0.08, 0.20, 0.70],
                ],
                dtype=float,
            )

    def fit(self, values: np.ndarray) -> "FakeGaussianHMM":
        FakeGaussianHMM.fit_calls += 1
        FakeGaussianHMM.last_fit_rows = int(values.shape[0])
        base = np.linspace(-1.0, 1.0, self.n_components).reshape(-1, 1)
        self.means_ = np.repeat(base, values.shape[1], axis=1)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        rows = []
        for _ in range(values.shape[0]):
            if self.n_components == 3:
                rows.append([0.26, 0.44, 0.30])
            else:
                rows.append([0.18, 0.24, 0.33, 0.25])
        return np.asarray(rows, dtype=float)


class DegenerateGaussianHMM(FakeGaussianHMM):
    def __init__(self, *, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        if self.n_components == 3:
            self.transmat_ = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        else:
            self.transmat_ = np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )


def _observation(days: int = 80, *, as_of: str = "2026-06-09T20:00:00Z") -> ObservationRecord:
    spy = [520.0 + (index * 1.2) + math.sin(index / 4.0) for index in range(days)]
    vix = [14.0 + (index * 0.08) + abs(math.sin(index / 5.0)) for index in range(days)]
    vvix = [88.0 + (index * 0.18) + abs(math.cos(index / 6.0)) for index in range(days)]
    vix9d = [13.2 + (index * 0.07) + abs(math.sin(index / 7.0)) for index in range(days)]
    vix3m = [16.4 + (index * 0.05) + abs(math.cos(index / 8.0)) for index in range(days)]
    return ObservationRecord(
        schema_version="observation.v1",
        as_of=as_of,
        source="unit_test",
        symbols={
            "SPY": {"last": spy[-1]},
            "VIX": {"last": vix[-1]},
            "VVIX": {"last": vvix[-1]},
            "VIX9D": {"last": vix9d[-1]},
            "VIX3M": {"last": vix3m[-1]},
        },
        history={
            "SPY_close": spy,
            "VIX": vix,
            "VVIX": vvix,
            "VIX9D": vix9d,
            "VIX3M": vix3m,
        },
        quality={"is_complete": True, "warnings": [], "stale_fields": []},
        provider_metadata={},
    )


def _feature_record(observation: ObservationRecord) -> FeatureRecord:
    from agentic_vol_regime_app.features.build_features import compute_feature_record

    return compute_feature_record(
        observation,
        feature_config={
            "feature_set_version": "vol_regime_features_v1",
            "lookback_windows": {
                "zscore_short": 22,
                "rv_short": 5,
                "rv_medium": 21,
                "drawdown_window": 21,
            },
        },
    )


def test_hmm_feature_validation_warns_when_history_is_insufficient(tmp_path: Path) -> None:
    original = hmm_belief.GaussianHMM
    hmm_belief.GaussianHMM = FakeGaussianHMM
    observation = _observation(days=18)
    feature_record = _feature_record(observation)
    try:
        record = compute_hmm_belief_record(
            observation,
            feature_record,
            app_paths=AppPaths(root=tmp_path),
            config=HMMConfig(),
        )
    finally:
        hmm_belief.GaussianHMM = original

    assert record.warnings
    assert "Insufficient" in record.warnings[0]


def test_hmm_training_uses_only_pre_as_of_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    FakeGaussianHMM.fit_calls = 0
    observation = _observation()
    feature_record = _feature_record(observation)
    config = HMMConfig(train_window=50)
    rows, _ = _build_historical_feature_rows(observation, config=config)

    compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=config,
    )

    expected_training_rows = min(len(rows) - 1, config.train_window)
    assert FakeGaussianHMM.last_fit_rows == expected_training_rows


def test_hmm_training_produces_valid_probability_vectors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    FakeGaussianHMM.fit_calls = 0
    observation = _observation()
    feature_record = _feature_record(observation)
    record = compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=HMMConfig(train_window=45),
    )

    assert abs(sum(record.state_probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= record.confidence <= 1.0
    assert record.emission_top_state in record.emission_state_probabilities
    assert set(record.persistence_lift) == set(record.state_probabilities)
    assert record.state_feature_summaries
    assert record.interpretation_notes


def test_hmm_transition_matrix_rows_sum_to_one(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    FakeGaussianHMM.fit_calls = 0
    record = compute_hmm_belief_record(
        _observation(),
        _feature_record(_observation()),
        app_paths=AppPaths(root=tmp_path),
        config=HMMConfig(train_window=45),
    )

    for row in record.transition_matrix:
        assert abs(sum(row) - 1.0) < 1e-6


def test_expected_duration_calculation_handles_near_persistent_states() -> None:
    assert _safe_expected_duration(0.0) == 1.0
    assert _safe_expected_duration(0.5) == 2.0
    assert _safe_expected_duration(0.999) == 999.0


def test_matrix_power_transition_calculations() -> None:
    matrix = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    powered = _matrix_power(matrix, 5)
    assert powered.shape == (2, 2)
    for row in powered:
        assert abs(float(np.sum(row)) - 1.0) < 1e-6


def test_transition_matrix_repair_handles_zero_rows() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.4],
        ],
        dtype=float,
    )

    repaired, warnings = _repair_transition_matrix(matrix)

    assert abs(float(np.sum(repaired[2])) - 1.0) < 1e-6
    assert repaired[2][2] == 1.0
    assert warnings


def test_degenerate_hmm_fit_returns_warning_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", DegenerateGaussianHMM)
    observation = _observation()
    feature_record = _feature_record(observation)

    record = compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=HMMConfig(train_window=45),
    )

    assert record.is_trained is False
    assert record.training_status == "degenerate_fit"
    assert "degenerate" in " ".join(record.warnings).lower() or "unused" in " ".join(record.warnings).lower()


def test_hmm_does_not_retrain_within_24_hours(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    FakeGaussianHMM.fit_calls = 0
    config = HMMConfig(train_window=45, min_retrain_interval_hours=24)

    observation = _observation(as_of="2026-06-09T20:00:00Z")
    feature_record = _feature_record(observation)
    compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=config,
    )

    observation_later = _observation(as_of="2026-06-09T20:05:00Z")
    feature_record_later = _feature_record(observation_later)
    compute_hmm_belief_record(
        observation_later,
        feature_record_later,
        app_paths=AppPaths(root=tmp_path),
        config=config,
    )

    assert FakeGaussianHMM.fit_calls == 1


def test_hmm_report_renders_section() -> None:
    feature_record = FeatureRecord(
        schema_version="features.v1",
        as_of="2026-06-09T20:00:00Z",
        feature_set_version="vol_regime_features_v1",
        features={"vix": 16.2, "vvix": 92.0, "vvix_vix_ratio": 5.68, "term_structure_state": "contango", "term_structure_symbol": "VIX3M", "realized_vol_acceleration": 0.01},
        missing_features=[],
        lookback_windows={},
    )
    belief_record = BeliefRecord(
        schema_version="belief.v1",
        as_of="2026-06-09T20:00:00Z",
        model_version="belief_model_v1",
        beliefs={
            "STABLE_LOW_VOL_TREND": 0.5,
            "MID_VOL_CHOP": 0.2,
            "VOL_EXPANSION_TRANSITION": 0.15,
            "HIGH_VOL_RISK_OFF": 0.1,
            "PANIC_CONVEXITY_STRESS": 0.03,
            "POST_PANIC_COMPRESSION": 0.02,
        },
        belief_delta={},
        entropy=0.2,
        confidence=0.72,
        drivers=[],
    )
    transition_record = TransitionProbabilityRecord(
        schema_version="transition.v1",
        as_of="2026-06-09T20:00:00Z",
        model_version="transition.v1",
        transition_probabilities={"vol_expansion_5d": 0.21},
        top_predictive_factors=[],
        confirming_features_count=2,
    )
    hmm_record = HMMBeliefRecord(
        schema_version="hmm_belief.v1",
        model_name="HMMBeliefAgent",
        model_version="hmm_gaussian_v1",
        as_of="2026-06-09T20:00:00Z",
        is_trained=True,
        training_status="trained",
        state_probabilities={
            "STABLE": 0.4,
            "EXPANDING_VOL": 0.35,
            "HIGH_VOL": 0.25,
        },
        top_state="STABLE",
        transition_matrix=[[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.03, 0.17, 0.8]],
        expected_duration_days={"STABLE": 5.0, "EXPANDING_VOL": 3.3, "HIGH_VOL": 5.0},
        current_state_expected_duration_days=5.0,
        persistence_probabilities={"current_state_5d": 0.33, "current_state_10d": 0.18, "current_state_21d": 0.07},
        transition_probabilities={
            "to_high_vol_stress_5d": 0.11,
            "to_high_vol_stress_10d": 0.14,
            "to_high_vol_stress_21d": 0.19,
            "to_vol_expansion_or_high_vol_5d": 0.32,
            "to_vol_expansion_or_high_vol_10d": 0.38,
            "to_vol_expansion_or_high_vol_21d": 0.44,
        },
        confidence=0.68,
        warnings=[],
        drivers=[],
        interpretation_notes=["Current features themselves fit STABLE best."],
        emission_state_probabilities={
            "STABLE": 0.43,
            "EXPANDING_VOL": 0.31,
            "HIGH_VOL": 0.26,
        },
        emission_top_state="STABLE",
        persistence_lift={
            "STABLE": -0.03,
            "EXPANDING_VOL": 0.04,
            "HIGH_VOL": -0.01,
        },
        state_feature_summaries={
            "STABLE": {
                "vix": 14.3,
                "realized_vol_21d": 11.8,
                "drawdown_21d": 0.01,
                "term_structure_slope": 2.4,
                "trend_persistence_21d": 0.71,
                "vvix_vix_ratio": 5.5,
            }
        },
    )
    alert_record = AlertRecord(
        schema_version="alert.v1",
        alert_id="alert",
        as_of="2026-06-09T20:00:00Z",
        severity="WATCH",
        alert_type="transition",
        headline="Watch transition risk.",
        probabilities={},
        belief_state={},
        drivers=[],
        recommended_review=[],
        requires_human_review=False,
    )
    policy_record = PolicyRecommendationRecord(
        schema_version="policy_recommendation.v1",
        as_of="2026-06-09T20:00:00Z",
        recommended_action="NO_OVERWRITE",
        confidence=0.71,
        rationale=["Stable structure still dominates."],
        risk_notes=["None"],
        requires_human_review=False,
    )
    critic_record = CriticReviewRecord(
        schema_version="critic_review.v1",
        as_of="2026-06-09T20:00:00Z",
        verdict="ALLOW",
        findings=["Looks coherent."],
        requires_human_review=False,
        summary="Looks coherent.",
    )
    markdown = render_daily_markdown(
        feature_record=feature_record,
        belief_record=belief_record,
        transition_record=transition_record,
        hmm_record=hmm_record,
        alert_record=alert_record,
        policy_record=policy_record,
        critic_record=critic_record,
        comparison_panel=[
            {"engine": "Heuristic", "top_regime": "Stable Low-Vol Trend", "confidence": 0.72, "recommended_posture": "NO_OVERWRITE"},
            {"engine": "Linear ML", "top_regime": "Stable Low-Vol Trend", "confidence": 0.66, "recommended_posture": "LIGHT_OVERWRITE"},
            {"engine": "HMM", "top_regime": "STABLE", "confidence": 0.68, "recommended_posture": "NO_OVERWRITE"},
            {"engine": "Ensemble (disabled)", "top_regime": "Disabled", "confidence": 0.0, "recommended_posture": "Disabled"},
        ],
    )

    assert "HMM Regime Persistence" in markdown
    assert "Belief Reconciliation" in markdown
    assert "Current-state expected duration" in markdown
    assert "Emission vs Persistence" in markdown
    assert "State Summaries" in markdown
    assert "Interpretation Notes" in markdown


def test_hmm_to_belief_record_maps_four_states_into_global_beliefs() -> None:
    hmm_record = HMMBeliefRecord(
        schema_version="hmm_belief.v1",
        model_name="HMMBeliefAgent",
        model_version="hmm_gaussian_v1",
        as_of="2026-06-09T20:00:00Z",
        is_trained=True,
        training_status="trained",
        state_probabilities={"STABLE": 0.4, "EXPANDING_VOL": 0.35, "HIGH_VOL": 0.25},
        top_state="EXPANDING_VOL",
        transition_matrix=[],
        expected_duration_days={},
        current_state_expected_duration_days=0.0,
        persistence_probabilities={},
        transition_probabilities={},
        confidence=0.67,
        warnings=[],
        drivers=[],
        interpretation_notes=[],
        emission_state_probabilities={"STABLE": 0.35, "EXPANDING_VOL": 0.4, "HIGH_VOL": 0.25},
        emission_top_state="EXPANDING_VOL",
        persistence_lift={"STABLE": 0.05, "EXPANDING_VOL": -0.05, "HIGH_VOL": 0.0},
        state_feature_summaries={},
    )

    belief = hmm_to_belief_record(hmm_record)

    assert abs(sum(belief.beliefs.values()) - 1.0) < 1e-6
    assert belief.beliefs["VOL_EXPANSION_TRANSITION"] == 0.35
