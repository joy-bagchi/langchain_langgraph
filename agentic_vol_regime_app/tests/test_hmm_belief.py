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
from agentic_vol_regime_app.features.sector_geometry import (
    SECTOR_ETF_UNIVERSE,
    build_sector_return_matrix,
    compute_sector_geometry_metrics,
)
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
    sector_history = {
        f"{symbol}_close": [80.0 + (idx * 0.15) + math.sin((idx / 6.0) + position) for idx in range(days)]
        for position, symbol in enumerate(SECTOR_ETF_UNIVERSE)
    }
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
            **sector_history,
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


def test_sector_geometry_metrics_return_expected_ranges() -> None:
    history = _observation(days=80).history
    metrics, warnings = compute_sector_geometry_metrics(history, lookback_days=21)

    assert warnings == []
    assert -1.0 <= metrics["avg_pairwise_corr_21d"] <= 1.0
    assert 0.0 <= metrics["first_eigenvalue_share_21d"] <= 1.0
    assert metrics["effective_rank_21d"] >= 1.0


def test_perfectly_correlated_synthetic_sectors_raise_market_mode_share() -> None:
    history: dict[str, list[float]] = {}
    base = [100.0 + float(index) for index in range(30)]
    for symbol in SECTOR_ETF_UNIVERSE:
        history[f"{symbol}_close"] = list(base)

    metrics, warnings = compute_sector_geometry_metrics(history, lookback_days=21)

    assert warnings == []
    assert 0.0 <= metrics["first_eigenvalue_share_21d"] <= 1.0
    assert metrics["first_eigenvalue_share_21d"] > 0.9


def test_independent_synthetic_sectors_lower_market_mode_share() -> None:
    history: dict[str, list[float]] = {}
    for position, symbol in enumerate(SECTOR_ETF_UNIVERSE):
        history[f"{symbol}_close"] = [
            100.0
            + (index * (1.0 + position * 0.03))
            + math.sin((index / 3.0) + position) * (4.0 + position * 0.1)
            + math.cos((index / 5.0) + position * 0.5) * 2.0
            for index in range(30)
        ]

    metrics, warnings = compute_sector_geometry_metrics(history, lookback_days=21)

    assert warnings == []
    assert 0.0 <= metrics["first_eigenvalue_share_21d"] <= 1.0
    assert metrics["first_eigenvalue_share_21d"] < 0.6


def test_effective_rank_decreases_when_sectors_become_highly_correlated() -> None:
    independent_history: dict[str, list[float]] = {}
    correlated_history: dict[str, list[float]] = {}
    base = [100.0 + float(index) for index in range(30)]
    for position, symbol in enumerate(SECTOR_ETF_UNIVERSE):
        independent_history[f"{symbol}_close"] = [
            100.0
            + (index * (1.0 + position * 0.03))
            + math.sin((index / 3.0) + position) * (4.0 + position * 0.1)
            + math.cos((index / 5.0) + position * 0.5) * 2.0
            for index in range(30)
        ]
        correlated_history[f"{symbol}_close"] = [value + (position * 0.01) for value in base]

    independent_metrics, _ = compute_sector_geometry_metrics(independent_history, lookback_days=21)
    correlated_metrics, _ = compute_sector_geometry_metrics(correlated_history, lookback_days=21)

    assert correlated_metrics["effective_rank_21d"] < independent_metrics["effective_rank_21d"]


def test_log_det_corr_falls_when_correlation_matrix_nears_singularity() -> None:
    independent_history: dict[str, list[float]] = {}
    near_singular_history: dict[str, list[float]] = {}
    base = [100.0 + float(index) for index in range(30)]
    for position, symbol in enumerate(SECTOR_ETF_UNIVERSE):
        independent_history[f"{symbol}_close"] = [
            100.0
            + (index * (1.0 + position * 0.03))
            + math.sin((index / 3.0) + position) * (4.0 + position * 0.1)
            + math.cos((index / 5.0) + position * 0.5) * 2.0
            for index in range(30)
        ]
        near_singular_history[f"{symbol}_close"] = [value + (position * 0.0001) for value in base]

    independent_metrics, _ = compute_sector_geometry_metrics(independent_history, lookback_days=21)
    near_singular_metrics, _ = compute_sector_geometry_metrics(near_singular_history, lookback_days=21)

    assert near_singular_metrics["log_det_corr_21d"] < independent_metrics["log_det_corr_21d"]


def test_sector_return_matrix_warns_when_symbols_are_missing() -> None:
    history = _observation(days=40).history
    for symbol in SECTOR_ETF_UNIVERSE[:2]:
        history.pop(f"{symbol}_close", None)

    matrix, warnings = build_sector_return_matrix(history, lookback_days=21)

    assert matrix is None
    assert warnings
    assert "Missing sector history" in warnings[0]


def test_hmm_v2_uses_sector_correlation_features(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    observation = _observation(days=120)
    feature_record = _feature_record(observation)
    config = hmm_belief.load_hmm_config(app_paths=AppPaths(root=tmp_path), variant_id="v2")
    record = compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=config,
        variant_id="v2",
    )

    assert record.variant_id == "v2"
    assert record.variant_label == "HMM v2 Core + Sector Corr"
    assert "avg_pairwise_corr_21d" in record.inference_feature_vector
    assert "first_eigenvalue_share_21d" in record.inference_feature_vector
    assert "avg_pairwise_corr_21d" in record.sector_metrics


def test_hmm_v3_uses_sector_geometry_features(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    observation = _observation(days=120)
    feature_record = _feature_record(observation)
    config = hmm_belief.load_hmm_config(app_paths=AppPaths(root=tmp_path), variant_id="v3")
    record = compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=config,
        variant_id="v3",
    )

    assert record.variant_id == "v3"
    assert record.variant_label == "HMM v3 Core + Geometry"
    assert "avg_pairwise_corr_21d" in record.inference_feature_vector
    assert "first_eigenvalue_share_21d" in record.inference_feature_vector
    assert "effective_rank_21d" in record.inference_feature_vector
    assert "log_det_corr_21d" in record.inference_feature_vector


def test_hmm_v3_falls_back_to_v1_when_sector_geometry_inputs_are_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hmm_belief, "GaussianHMM", FakeGaussianHMM)
    observation = _observation(days=120)
    for symbol in SECTOR_ETF_UNIVERSE:
        observation.history.pop(f"{symbol}_close", None)
    feature_record = _feature_record(observation)
    config = hmm_belief.load_hmm_config(app_paths=AppPaths(root=tmp_path), variant_id="v3")

    record = compute_hmm_belief_record(
        observation,
        feature_record,
        app_paths=AppPaths(root=tmp_path),
        config=config,
        variant_id="v3",
    )

    assert record.variant_id == "v1"
    assert any("fell back to hmm v1" in warning.lower() for warning in record.warnings)


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
            "LOW_VOL_TREND": 0.3,
            "MID_VOL_CHOP": 0.1,
            "VOL_EXPANSION": 0.35,
            "HIGH_VOL_STRESS": 0.25,
        },
        top_state="LOW_VOL_TREND",
        transition_matrix=[[0.8, 0.15, 0.03, 0.02], [0.1, 0.65, 0.15, 0.1], [0.03, 0.17, 0.65, 0.15], [0.02, 0.08, 0.2, 0.7]],
        expected_duration_days={"LOW_VOL_TREND": 5.0, "MID_VOL_CHOP": 2.9, "VOL_EXPANSION": 2.9, "HIGH_VOL_STRESS": 3.3},
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
        interpretation_notes=["Current features themselves fit LOW_VOL_TREND best."],
        emission_state_probabilities={
            "LOW_VOL_TREND": 0.34,
            "MID_VOL_CHOP": 0.12,
            "VOL_EXPANSION": 0.31,
            "HIGH_VOL_STRESS": 0.23,
        },
        emission_top_state="LOW_VOL_TREND",
        persistence_lift={
            "LOW_VOL_TREND": -0.04,
            "MID_VOL_CHOP": -0.02,
            "VOL_EXPANSION": 0.04,
            "HIGH_VOL_STRESS": 0.02,
        },
        state_feature_summaries={
            "LOW_VOL_TREND": {
                "vix": 14.3,
                "realized_vol_21d": 11.8,
                "drawdown_21d": 0.01,
                "term_structure_slope": 2.4,
                "trend_persistence_21d": 0.71,
                "vvix_vix_ratio": 5.5,
            }
        },
        sector_metrics={
            "avg_pairwise_corr_21d": 0.42,
            "first_eigenvalue_share_21d": 0.53,
            "effective_rank_21d": 2.81,
            "log_det_corr_21d": -3.12,
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
        hmm_variant_comparison=[
            {
                "model": "HMM v1 Core",
                "top_state": "LOW_VOL_TREND",
                "confidence": 0.68,
                "expected_duration_days": 5.0,
                "transition_10d_high_vol_prob": 0.14,
                "recommendation": "NO_OVERWRITE",
            },
            {
                "model": "HMM v2 Core + Sector Corr",
                "top_state": "VOL_EXPANSION",
                "confidence": 0.72,
                "expected_duration_days": 4.1,
                "transition_10d_high_vol_prob": 0.22,
                "recommendation": "LIGHT_OVERWRITE",
            },
            {
                "model": "HMM v3 Core + Geometry",
                "top_state": "HIGH_VOL_STRESS",
                "confidence": 0.77,
                "expected_duration_days": 3.4,
                "transition_10d_high_vol_prob": 0.31,
                "recommendation": "MEDIUM_OVERWRITE",
            },
        ],
    )

    assert "HMM Regime Persistence" in markdown
    assert "Current-state expected duration" in markdown
    assert "Model Variant Comparison" in markdown
    assert "Sector Correlation / Market Mode" in markdown


def test_hmm_to_belief_record_maps_four_states_into_global_beliefs() -> None:
    hmm_record = HMMBeliefRecord(
        schema_version="hmm_belief.v1",
        model_name="HMMBeliefAgent",
        model_version="hmm_gaussian_v1",
        as_of="2026-06-09T20:00:00Z",
        is_trained=True,
        training_status="trained",
        state_probabilities={"LOW_VOL_TREND": 0.25, "MID_VOL_CHOP": 0.15, "VOL_EXPANSION": 0.35, "HIGH_VOL_STRESS": 0.25},
        top_state="VOL_EXPANSION",
        transition_matrix=[],
        expected_duration_days={},
        current_state_expected_duration_days=0.0,
        persistence_probabilities={},
        transition_probabilities={},
        confidence=0.67,
        warnings=[],
        drivers=[],
        interpretation_notes=[],
        emission_state_probabilities={"LOW_VOL_TREND": 0.2, "MID_VOL_CHOP": 0.2, "VOL_EXPANSION": 0.4, "HIGH_VOL_STRESS": 0.2},
        emission_top_state="VOL_EXPANSION",
        persistence_lift={"LOW_VOL_TREND": 0.05, "MID_VOL_CHOP": -0.05, "VOL_EXPANSION": -0.05, "HIGH_VOL_STRESS": 0.05},
        state_feature_summaries={},
    )

    belief = hmm_to_belief_record(hmm_record)

    assert abs(sum(belief.beliefs.values()) - 1.0) < 1e-6
    assert belief.beliefs["VOL_EXPANSION_TRANSITION"] == 0.35
