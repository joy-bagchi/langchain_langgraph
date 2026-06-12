"""Hidden Markov Model backed belief-state engine for volatility regimes."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional runtime dependency
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    GaussianHMM = None  # type: ignore[assignment]

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.contracts import BeliefRecord, FeatureRecord, HMMBeliefRecord, ObservationRecord
from agentic_vol_regime_app.features.sector_geometry import compute_sector_geometry_metrics


HMM_STATE_ORDER = (
    "STABLE",
    "EXPANDING_VOL",
    "HIGH_VOL",
)

HMM_TO_GLOBAL_REGIME = {
    "STABLE": "STABLE_LOW_VOL_TREND",
    "EXPANDING_VOL": "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL": "HIGH_VOL_RISK_OFF",
}

DEFAULT_FEATURES = [
    "spy_return_1d",
    "realized_vol_5d",
    "realized_vol_21d",
    "vix",
    "vix_z_22d",
    "vvix",
    "vvix_vix_ratio",
    "vvix_vix_z_22d",
    "vix9d_vix_ratio",
    "vix_vix3m_ratio",
    "term_structure_slope",
    "drawdown_21d",
    "trend_persistence_21d",
]

SECTOR_CORR_FEATURES = [
    "avg_pairwise_corr_21d",
    "first_eigenvalue_share_21d",
]


@dataclass(frozen=True, slots=True)
class HMMVariantSpec:
    variant_id: str
    model_version: str
    config_name: str
    artifact_name: str
    label: str
    feature_flags: dict[str, bool]


HMM_VARIANTS: dict[str, HMMVariantSpec] = {
    "v1": HMMVariantSpec(
        variant_id="v1",
        model_version="hmm_gaussian_v1",
        config_name="hmm_v1_core.yaml",
        artifact_name="daily_regime_hmm_v1_model.pkl",
        label="HMM v1 Core",
        feature_flags={
            "enable_hmm_v2_sector_corr": False,
            "enable_hmm_v3_sector_geometry": False,
            "enable_sector_geometry_only_diagnostic": False,
        },
    ),
    "v2": HMMVariantSpec(
        variant_id="v2",
        model_version="hmm_gaussian_v2",
        config_name="hmm_v2_core_plus_sector_corr.yaml",
        artifact_name="daily_regime_hmm_v2_model.pkl",
        label="HMM v2 Core + Sector Corr",
        feature_flags={
            "enable_hmm_v2_sector_corr": True,
            "enable_hmm_v3_sector_geometry": False,
            "enable_sector_geometry_only_diagnostic": False,
        },
    ),
}


@dataclass(slots=True)
class HMMConfig:
    n_components: int = 3
    covariance_type: str = "diag"
    n_iter: int = 500
    random_state: int = 17
    feature_list: list[str] = None  # type: ignore[assignment]
    train_window: int = 756
    retrain_cadence: int = 5
    min_retrain_interval_hours: int = 24
    variant_id: str = "v1"
    variant_label: str = "HMM v1 Core"
    model_version: str = "hmm_gaussian_v1"
    model_artifact_name: str = "daily_regime_hmm_v1_model.pkl"
    feature_flags: dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.feature_list is None:
            self.feature_list = list(DEFAULT_FEATURES)
        if self.feature_flags is None:
            self.feature_flags = dict(HMM_VARIANTS[self.variant_id].feature_flags)


def _resolve_variant_spec(variant_id: str | None) -> HMMVariantSpec:
    normalized = str(variant_id or "v1").strip().lower()
    return HMM_VARIANTS.get(normalized, HMM_VARIANTS["v1"])


def load_hmm_config(*, app_paths: AppPaths, variant_id: str = "v1") -> HMMConfig:
    spec = _resolve_variant_spec(variant_id)
    config_path = app_paths.features_dir / spec.config_name
    legacy_v1_path = app_paths.features_dir / "hmm_model_v1.yaml"
    if not config_path.exists() and spec.variant_id == "v1" and legacy_v1_path.exists():
        config_path = legacy_v1_path
    if not config_path.exists():
        return HMMConfig(
            feature_list=list(DEFAULT_FEATURES) + (SECTOR_CORR_FEATURES if spec.variant_id == "v2" else []),
            variant_id=spec.variant_id,
            variant_label=spec.label,
            model_version=spec.model_version,
            model_artifact_name=spec.artifact_name,
            feature_flags=dict(spec.feature_flags),
        )

    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default_feature_list = list(DEFAULT_FEATURES) + (SECTOR_CORR_FEATURES if spec.variant_id == "v2" else [])
    return HMMConfig(
        n_components=int(payload.get("n_components", 3)),
        covariance_type=str(payload.get("covariance_type", "diag")),
        n_iter=int(payload.get("n_iter", 500)),
        random_state=int(payload.get("random_state", 17)),
        feature_list=list(payload.get("feature_list", default_feature_list)),
        train_window=int(payload.get("train_window", 756)),
        retrain_cadence=int(payload.get("retrain_cadence", 5)),
        min_retrain_interval_hours=int(payload.get("min_retrain_interval_hours", 24)),
        variant_id=str(payload.get("variant_id", spec.variant_id)),
        variant_label=str(payload.get("variant_label", spec.label)),
        model_version=str(payload.get("model_version", spec.model_version)),
        model_artifact_name=str(payload.get("model_artifact_name", spec.artifact_name)),
        feature_flags={
            **dict(spec.feature_flags),
            **dict(payload.get("feature_flags", {})),
        },
    )


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _zscore_for_index(series: list[float], index: int, lookback: int) -> float | None:
    if index < lookback - 1:
        return None
    window = series[index - lookback + 1 : index + 1]
    sigma = _sample_std(window)
    if sigma <= 1e-9:
        return 0.0
    mean = sum(window) / len(window)
    return (series[index] - mean) / sigma


def _annualized_realized_vol_at(closes: list[float], index: int, lookback: int) -> float | None:
    if index < lookback:
        return None
    window = closes[index - lookback : index + 1]
    returns: list[float] = []
    for current_index in range(1, len(window)):
        previous = window[current_index - 1]
        current = window[current_index]
        if previous <= 0 or current <= 0:
            return None
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return None
    return _sample_std(returns) * math.sqrt(252.0) * 100.0


def _drawdown_at(closes: list[float], index: int, lookback: int) -> float | None:
    if index < lookback - 1:
        return None
    window = closes[index - lookback + 1 : index + 1]
    peak = max(window)
    if peak <= 0:
        return None
    return max(0.0, (peak - window[-1]) / peak)


def _trend_persistence_at(closes: list[float], index: int, lookback: int) -> float | None:
    if index < lookback:
        return None
    window = closes[index - lookback : index + 1]
    positive_steps = 0
    total_steps = 0
    for current_index in range(1, len(window)):
        total_steps += 1
        if window[current_index] >= window[current_index - 1]:
            positive_steps += 1
    if total_steps == 0:
        return None
    return positive_steps / total_steps


def _build_sector_history_window(
    observation: ObservationRecord,
    *,
    end_index: int,
    window_length: int,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    start_index = end_index - window_length + 1
    for key, values in observation.history.items():
        if key.endswith("_close") and key.startswith("XL"):
            numeric = [float(item) for item in values if item is not None]
            if len(numeric) > end_index:
                history[key] = numeric[start_index : end_index + 1]
    return history


def _build_historical_feature_rows(
    observation: ObservationRecord,
    *,
    config: HMMConfig,
) -> tuple[list[dict[str, float]], list[str]]:
    spy_closes = [float(value) for value in observation.history.get("SPY_close", []) if value is not None]
    vix_history = [float(value) for value in observation.history.get("VIX", []) if value is not None]
    vvix_history = [float(value) for value in observation.history.get("VVIX", []) if value is not None]
    vix9d_history = [float(value) for value in observation.history.get("VIX9D", []) if value is not None]
    vix3m_history = [float(value) for value in observation.history.get("VIX3M", []) if value is not None]

    min_length = min(len(spy_closes), len(vix_history), len(vvix_history), len(vix9d_history), len(vix3m_history))
    warnings: list[str] = []
    if min_length < 25:
        warnings.append("Insufficient aligned history for HMM training or inference.")
        return [], warnings

    spy_closes = spy_closes[-min_length:]
    vix_history = vix_history[-min_length:]
    vvix_history = vvix_history[-min_length:]
    vix9d_history = vix9d_history[-min_length:]
    vix3m_history = vix3m_history[-min_length:]

    rows: list[dict[str, float]] = []
    sector_warning_seen = False
    for index in range(min_length):
        if index < 22:
            continue
        vix = vix_history[index]
        vvix = vvix_history[index]
        vix9d = vix9d_history[index]
        vix3m = vix3m_history[index]
        spy = spy_closes[index]
        prior_spy = spy_closes[index - 1] if index >= 1 else None
        rv_5d = _annualized_realized_vol_at(spy_closes, index, 5)
        rv_21d = _annualized_realized_vol_at(spy_closes, index, 21)
        vix_z = _zscore_for_index(vix_history, index, 22)
        ratio_history = [
            vvix_history[item] / vix_history[item]
            for item in range(index - 21, index + 1)
            if vix_history[item] > 0
        ]
        vvix_vix_ratio = vvix / vix if vix > 0 else None
        vvix_vix_z = None
        if vvix_vix_ratio is not None and len(ratio_history) >= 2:
            sigma = _sample_std(ratio_history)
            if sigma <= 1e-9:
                vvix_vix_z = 0.0
            else:
                vvix_vix_z = (vvix_vix_ratio - (sum(ratio_history) / len(ratio_history))) / sigma
        drawdown = _drawdown_at(spy_closes, index, 21)
        trend_persistence = _trend_persistence_at(spy_closes, index, 21)
        term_slope = vix3m - vix

        values: dict[str, float | None] = {
            "spy_return_1d": ((spy / prior_spy) - 1.0) if prior_spy and prior_spy > 0 else None,
            "realized_vol_5d": rv_5d,
            "realized_vol_21d": rv_21d,
            "vix": vix,
            "vix_z_22d": vix_z,
            "vvix": vvix,
            "vvix_vix_ratio": vvix_vix_ratio,
            "vvix_vix_z_22d": vvix_vix_z,
            "vix9d_vix_ratio": (vix9d / vix) if vix > 0 else None,
            "vix_vix3m_ratio": (vix / vix3m) if vix3m > 0 else None,
            "term_structure_slope": term_slope,
            "drawdown_21d": drawdown,
            "trend_persistence_21d": trend_persistence,
        }
        if config.feature_flags.get("enable_hmm_v2_sector_corr", False):
            sector_history = _build_sector_history_window(observation, end_index=index, window_length=21)
            sector_metrics, sector_warnings = compute_sector_geometry_metrics(sector_history, lookback_days=21)
            if sector_warnings and not sector_warning_seen:
                warnings.extend(sector_warnings)
                sector_warning_seen = True
            values["avg_pairwise_corr_21d"] = sector_metrics.get("avg_pairwise_corr_21d")
            values["first_eigenvalue_share_21d"] = sector_metrics.get("first_eigenvalue_share_21d")

        if any(values.get(name) is None for name in config.feature_list):
            continue
        rows.append({name: float(values[name]) for name in config.feature_list if values.get(name) is not None})
    if len(rows) < 3:
        warnings.append("Insufficient fully populated feature rows for HMM training.")
    return rows, warnings


def _matrix_from_rows(rows: list[dict[str, float]], feature_list: list[str]) -> np.ndarray:
    return np.asarray([[row[name] for name in feature_list] for row in rows], dtype=float)


def _model_artifact_path(*, app_paths: AppPaths, config: HMMConfig) -> Path:
    model_dir = app_paths.models_dir / "hmm"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / config.model_artifact_name


def _load_artifact(*, app_paths: AppPaths, config: HMMConfig) -> dict[str, Any] | None:
    path = _model_artifact_path(app_paths=app_paths, config=config)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_artifact(*, app_paths: AppPaths, config: HMMConfig, artifact: dict[str, Any]) -> None:
    path = _model_artifact_path(app_paths=app_paths, config=config)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def _should_retrain(
    artifact: dict[str, Any] | None,
    *,
    config: HMMConfig,
    training_row_count: int,
    as_of: str,
) -> bool:
    if artifact is None:
        return True
    previous_rows = int(artifact.get("training_row_count", 0))
    if previous_rows <= 0:
        return True
    if int(artifact.get("n_components", 0) or 0) != int(config.n_components):
        return True
    if list(artifact.get("feature_list", [])) != list(config.feature_list):
        return True
    if str(artifact.get("covariance_type", "")) != str(config.covariance_type):
        return True
    if str(artifact.get("model_version", "")) != str(config.model_version):
        return True
    current_ts = _parse_timestamp(as_of)
    last_trained_ts = _parse_timestamp(str(artifact.get("last_trained_at", "")))
    if current_ts is None or last_trained_ts is None:
        return abs(training_row_count - previous_rows) >= max(1, config.retrain_cadence)
    elapsed_hours = (current_ts - last_trained_ts).total_seconds() / 3600.0
    if elapsed_hours < float(max(1, config.min_retrain_interval_hours)):
        return False
    return True


def _rank_state_labels(means: np.ndarray, feature_list: list[str]) -> dict[int, str]:
    index_by_name = {name: idx for idx, name in enumerate(feature_list)}
    scored: list[tuple[int, float]] = []
    for raw_state in range(means.shape[0]):
        vector = means[raw_state]
        risk_score = (
            vector[index_by_name["vix"]]
            + vector[index_by_name["realized_vol_21d"]]
            + (vector[index_by_name["drawdown_21d"]] * 25.0)
            + vector[index_by_name["vvix_vix_ratio"]]
            - (vector[index_by_name["trend_persistence_21d"]] * 10.0)
            - vector[index_by_name["term_structure_slope"]]
        )
        if "first_eigenvalue_share_21d" in index_by_name:
            risk_score += vector[index_by_name["first_eigenvalue_share_21d"]] * 15.0
        if "avg_pairwise_corr_21d" in index_by_name:
            risk_score += vector[index_by_name["avg_pairwise_corr_21d"]] * 8.0
        scored.append((raw_state, float(risk_score)))
    ranked = [item[0] for item in sorted(scored, key=lambda item: item[1])]
    return {raw_state: label for raw_state, label in zip(ranked, list(HMM_STATE_ORDER))}


def _state_probabilities_from_raw(posterior: np.ndarray, mapping: dict[int, str]) -> dict[str, float]:
    values = {label: 0.0 for label in HMM_STATE_ORDER}
    for raw_state, probability in enumerate(posterior):
        label = mapping.get(raw_state, HMM_STATE_ORDER[min(raw_state, len(HMM_STATE_ORDER) - 1)])
        values[label] += float(probability)
    total = sum(values.values()) or 1.0
    return {label: round(values[label] / total, 6) for label in HMM_STATE_ORDER}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    total = float(np.sum(exps)) or 1.0
    return exps / total


def _emission_probabilities_from_raw(model: Any, scaled_vector: np.ndarray, mapping: dict[int, str]) -> dict[str, float]:
    if not hasattr(model, "_compute_log_likelihood"):
        return {label: 0.0 for label in HMM_STATE_ORDER}
    try:
        log_likelihood = np.asarray(model._compute_log_likelihood(scaled_vector), dtype=float)[0]  # type: ignore[attr-defined]
    except Exception:
        return {label: 0.0 for label in HMM_STATE_ORDER}
    posterior = _softmax(log_likelihood)
    return _state_probabilities_from_raw(posterior, mapping)


def _build_state_feature_summaries(
    means: np.ndarray,
    feature_list: list[str],
    mapping: dict[int, str],
) -> dict[str, dict[str, float]]:
    index_by_name = {name: idx for idx, name in enumerate(feature_list)}
    summary_features = [
        "vix",
        "realized_vol_21d",
        "drawdown_21d",
        "term_structure_slope",
        "trend_persistence_21d",
        "vvix_vix_ratio",
        "vix_vix3m_ratio",
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
    ]
    summaries: dict[str, dict[str, float]] = {}
    for raw_state in range(means.shape[0]):
        label = mapping.get(raw_state, HMM_STATE_ORDER[min(raw_state, len(HMM_STATE_ORDER) - 1)])
        vector = means[raw_state]
        summaries[label] = {
            name: round(float(vector[index_by_name[name]]), 6)
            for name in summary_features
            if name in index_by_name
        }
    return summaries


def _build_interpretation_notes(
    *,
    config: HMMConfig,
    top_state: str,
    emission_top_state: str,
    state_probabilities: dict[str, float],
    emission_state_probabilities: dict[str, float],
    persistence_lift: dict[str, float],
    transition_probabilities: dict[str, float],
    state_feature_summaries: dict[str, dict[str, float]],
    sector_metrics: dict[str, float],
) -> list[str]:
    notes: list[str] = []
    if top_state == emission_top_state:
        notes.append(f"Current features themselves fit `{top_state}` best; emission-only and path-aware posteriors agree.")
    else:
        notes.append(
            f"Current features fit `{emission_top_state}` best, but transition persistence lifts the final HMM call to `{top_state}`."
        )

    top_lift = persistence_lift.get(top_state, 0.0)
    emission_prob = emission_state_probabilities.get(top_state, 0.0)
    posterior_prob = state_probabilities.get(top_state, 0.0)
    notes.append(f"`{top_state}` posterior is {posterior_prob:.2f} vs emission-only {emission_prob:.2f}; persistence lift is {top_lift:+.2f}.")

    top_summary = state_feature_summaries.get(top_state, {})
    if top_summary:
        notes.append(
            "Mapped state summary: "
            f"VIX {top_summary.get('vix', 0.0):.2f}, "
            f"RV21 {top_summary.get('realized_vol_21d', 0.0):.2f}, "
            f"drawdown {top_summary.get('drawdown_21d', 0.0):.2f}, "
            f"slope {top_summary.get('term_structure_slope', 0.0):.2f}."
        )

    expansion_risk = transition_probabilities.get("to_vol_expansion_or_high_vol_5d", 0.0)
    stress_risk = transition_probabilities.get("to_high_vol_stress_5d", 0.0)
    if expansion_risk >= 0.35 or stress_risk >= 0.2:
        notes.append(
            f"Forward transition risk is elevated: 5d expansion/high-vol probability {expansion_risk:.2f}, high-vol stress probability {stress_risk:.2f}."
        )
    else:
        notes.append(
            f"Forward transition risk is moderate: 5d expansion/high-vol probability {expansion_risk:.2f}, high-vol stress probability {stress_risk:.2f}."
        )

    if config.feature_flags.get("enable_hmm_v2_sector_corr", False) and sector_metrics:
        avg_corr = sector_metrics.get("avg_pairwise_corr_21d", 0.0)
        first_share = sector_metrics.get("first_eigenvalue_share_21d", 0.0)
        if avg_corr < 0.35 and first_share < 0.35:
            notes.append("Sector correlations remain relatively independent; the market mode is not dominant.")
        else:
            notes.append("Sector co-movement is elevated; rising market-mode dominance raises vol-expansion risk.")
    return notes


def _safe_expected_duration(persistence: float) -> float:
    if persistence >= 0.999:
        return 999.0
    if persistence <= 0.0:
        return 1.0
    denominator = max(1e-6, 1.0 - persistence)
    return round(1.0 / denominator, 6)


def _state_order_indices(mapping: dict[int, str]) -> dict[str, int]:
    return {label: raw_state for raw_state, label in mapping.items()}


def _matrix_power(matrix: np.ndarray, n: int) -> np.ndarray:
    return np.linalg.matrix_power(matrix, n)


def _repair_transition_matrix(matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    repaired = np.asarray(matrix, dtype=float).copy()
    warnings: list[str] = []
    if repaired.ndim != 2 or repaired.shape[0] != repaired.shape[1]:
        return repaired, warnings
    for index in range(repaired.shape[0]):
        row = np.asarray(repaired[index], dtype=float)
        row = np.where(np.isfinite(row), row, 0.0)
        row = np.clip(row, 0.0, None)
        row_sum = float(np.sum(row))
        if row_sum <= 1e-12:
            row = np.zeros_like(row)
            row[index] = 1.0
            warnings.append(f"HMM transition row {index} had no usable mass and was repaired to a self-loop.")
        else:
            row = row / row_sum
            if abs(row_sum - 1.0) > 1e-6:
                warnings.append(f"HMM transition row {index} was renormalized from row sum {row_sum:.6f}.")
        repaired[index] = row
    return repaired, warnings


def _compute_state_usage_counts(model: Any, scaled_train: np.ndarray, mapping: dict[int, str]) -> dict[str, int]:
    if scaled_train.size == 0:
        return {label: 0 for label in HMM_STATE_ORDER}
    posterior = np.asarray(model.predict_proba(scaled_train), dtype=float)
    raw_assignments = np.argmax(posterior, axis=1)
    counts = {label: 0 for label in HMM_STATE_ORDER}
    for raw_state in raw_assignments:
        counts[mapping.get(int(raw_state), "STABLE")] += 1
    return counts


def _warning_record(
    *,
    as_of: str,
    config: HMMConfig,
    warnings: list[str],
    drivers: list[str] | None = None,
    training_status: str = "not_trained_enough",
) -> HMMBeliefRecord:
    zero_matrix = [[0.0 for _ in HMM_STATE_ORDER] for _ in HMM_STATE_ORDER]
    probabilities = {label: 0.0 for label in HMM_STATE_ORDER}
    return HMMBeliefRecord(
        schema_version="hmm_belief.v1",
        model_name="HMMBeliefAgent",
        model_version=config.model_version,
        as_of=as_of,
        is_trained=False,
        training_status=training_status,
        state_probabilities=probabilities,
        top_state="NOT_TRAINED_ENOUGH",
        transition_matrix=zero_matrix,
        expected_duration_days={label: 0.0 for label in HMM_STATE_ORDER},
        current_state_expected_duration_days=0.0,
        persistence_probabilities={
            "current_state_5d": 0.0,
            "current_state_10d": 0.0,
            "current_state_21d": 0.0,
        },
        transition_probabilities={
            "to_high_vol_stress_5d": 0.0,
            "to_high_vol_stress_10d": 0.0,
            "to_high_vol_stress_21d": 0.0,
            "to_vol_expansion_or_high_vol_5d": 0.0,
            "to_vol_expansion_or_high_vol_10d": 0.0,
            "to_vol_expansion_or_high_vol_21d": 0.0,
        },
        confidence=0.0,
        warnings=list(warnings),
        drivers=list(drivers or []),
        interpretation_notes=[str(warnings[0]) if warnings else "HMM explainability is unavailable because the advisory model did not run."],
        state_label_mapping={str(index): label for index, label in enumerate(HMM_STATE_ORDER)},
        emission_state_probabilities={label: 0.0 for label in HMM_STATE_ORDER},
        emission_top_state="NOT_TRAINED_ENOUGH",
        persistence_lift={label: 0.0 for label in HMM_STATE_ORDER},
        state_feature_summaries={},
        training_row_count=0,
        configured_train_window=0,
        inference_feature_vector={},
        variant_id=config.variant_id,
        variant_label=config.variant_label,
        model_converged=False,
        state_usage_counts={label: 0 for label in HMM_STATE_ORDER},
        sector_metrics={},
    )


def compute_hmm_belief_record(
    observation: ObservationRecord,
    feature_record: FeatureRecord,
    *,
    app_paths: AppPaths,
    config: HMMConfig | None = None,
    variant_id: str = "v1",
) -> HMMBeliefRecord:
    config = config or load_hmm_config(app_paths=app_paths, variant_id=variant_id)
    if GaussianHMM is None:
        return _warning_record(
            as_of=feature_record.as_of,
            config=config,
            warnings=["hmmlearn is not installed; HMM advisory output is unavailable."],
        )

    rows, warnings = _build_historical_feature_rows(observation, config=config)
    if not rows:
        if config.variant_id != "v1":
            fallback_record = compute_hmm_belief_record(
                observation,
                feature_record,
                app_paths=app_paths,
                config=load_hmm_config(app_paths=app_paths, variant_id="v1"),
                variant_id="v1",
            )
            fallback_warnings = list(dict.fromkeys(list(warnings) + list(fallback_record.warnings) + ["Fell back to HMM v1 because HMM v2 sector data was incomplete."]))
            fallback_record.warnings = fallback_warnings
            fallback_record.drivers = list(fallback_record.drivers) + ["HMM v2 sector features were unavailable, so the app reverted to the HMM v1 baseline."]
            return fallback_record
        return _warning_record(
            as_of=feature_record.as_of,
            config=config,
            warnings=warnings or ["No HMM feature rows could be constructed from the available history."],
        )

    latest_row = rows[-1]
    training_rows = rows[:-1]
    if len(training_rows) < max(2, config.n_components):
        if config.variant_id != "v1":
            return compute_hmm_belief_record(
                observation,
                feature_record,
                app_paths=app_paths,
                config=load_hmm_config(app_paths=app_paths, variant_id="v1"),
                variant_id="v1",
            )
        return _warning_record(
            as_of=feature_record.as_of,
            config=config,
            warnings=warnings + ["Insufficient pre-as-of rows to train the HMM without lookahead."],
            drivers=["The latest feature row was held out for inference, leaving too little history for training."],
        )

    training_rows = training_rows[-config.train_window :]
    feature_matrix = _matrix_from_rows(training_rows, config.feature_list)
    inference_matrix = _matrix_from_rows([latest_row], config.feature_list)
    artifact = _load_artifact(app_paths=app_paths, config=config)
    repair_warnings: list[str] = []
    if _should_retrain(artifact, config=config, training_row_count=len(training_rows), as_of=feature_record.as_of):
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(feature_matrix)
        model = GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            random_state=config.random_state,
        )
        model.fit(scaled_train)
        repaired_transmat, repair_warnings = _repair_transition_matrix(np.asarray(model.transmat_, dtype=float))
        model.transmat_ = repaired_transmat
        if any("had no usable mass" in warning for warning in repair_warnings):
            return _warning_record(
                as_of=feature_record.as_of,
                config=config,
                warnings=warnings + repair_warnings + ["HMM fit was rejected because one or more latent states were effectively unused."],
                drivers=["The current HMM fit produced a degenerate transition matrix, so the app fell back instead of trusting a brittle posterior."],
                training_status="degenerate_fit",
            )
        means = scaler.inverse_transform(model.means_)
        state_mapping = _rank_state_labels(np.asarray(means, dtype=float), config.feature_list)
        state_usage_counts = _compute_state_usage_counts(model, scaled_train, state_mapping)
        artifact = {
            "scaler": scaler,
            "model": model,
            "state_mapping": state_mapping,
            "state_usage_counts": state_usage_counts,
            "training_row_count": len(training_rows),
            "feature_list": list(config.feature_list),
            "n_components": config.n_components,
            "covariance_type": config.covariance_type,
            "model_version": config.model_version,
            "repair_warnings": list(repair_warnings),
            "last_trained_at": feature_record.as_of,
            "trained_as_of": feature_record.as_of,
            "train_window": config.train_window,
            "variant_id": config.variant_id,
            "variant_label": config.variant_label,
            "model_converged": bool(getattr(getattr(model, "monitor_", None), "converged", True)),
        }
        _save_artifact(app_paths=app_paths, config=config, artifact=artifact)
    else:
        scaler = artifact["scaler"]
        model = artifact["model"]
        state_mapping = dict(artifact["state_mapping"])
        state_usage_counts = {str(key): int(value) for key, value in dict(artifact.get("state_usage_counts", {})).items()}
        repaired_transmat, repair_warnings = _repair_transition_matrix(np.asarray(model.transmat_, dtype=float))
        model.transmat_ = repaired_transmat
        if any("had no usable mass" in warning for warning in repair_warnings):
            return _warning_record(
                as_of=feature_record.as_of,
                config=config,
                warnings=list(artifact.get("repair_warnings", [])) + repair_warnings + ["Stored HMM artifact is degenerate and was not used for inference."],
                drivers=["The cached HMM artifact contains an unusable transition row, so the app fell back instead of trusting a brittle posterior."],
                training_status="degenerate_fit",
            )
        means = scaler.inverse_transform(model.means_)

    scaled_all = scaler.transform(np.vstack([feature_matrix, inference_matrix]))
    posterior = np.asarray(model.predict_proba(scaled_all)[-1], dtype=float)
    state_probabilities = _state_probabilities_from_raw(posterior, state_mapping)
    top_state = max(state_probabilities, key=state_probabilities.get)
    emission_state_probabilities = _emission_probabilities_from_raw(model, scaled_all[-1:].copy(), state_mapping)
    if not any(emission_state_probabilities.values()):
        emission_state_probabilities = dict(state_probabilities)
    emission_top_state = max(emission_state_probabilities, key=emission_state_probabilities.get)
    persistence_lift = {
        label: round(float(state_probabilities.get(label, 0.0) - emission_state_probabilities.get(label, 0.0)), 6)
        for label in HMM_STATE_ORDER
    }
    state_feature_summaries = _build_state_feature_summaries(np.asarray(means, dtype=float), config.feature_list, state_mapping)

    transition_matrix = np.asarray(model.transmat_, dtype=float)
    transition_rows = transition_matrix.tolist()
    expected_duration_days = {
        state_mapping.get(index, HMM_STATE_ORDER[min(index, len(HMM_STATE_ORDER) - 1)]): _safe_expected_duration(float(transition_matrix[index][index]))
        for index in range(transition_matrix.shape[0])
    }
    label_to_index = _state_order_indices(state_mapping)
    current_raw_index = label_to_index.get(top_state, 0)
    current_state_expected_duration_days = expected_duration_days.get(top_state, 0.0)

    persistence_probabilities: dict[str, float] = {}
    transition_probabilities: dict[str, float] = {}
    high_vol_index = label_to_index.get("HIGH_VOL", current_raw_index)
    expansion_index = label_to_index.get("EXPANDING_VOL", current_raw_index)
    current_distribution = np.asarray([posterior], dtype=float)
    for horizon in (5, 10, 21):
        power = _matrix_power(transition_matrix, horizon)
        persistence_probabilities[f"current_state_{horizon}d"] = round(float(power[current_raw_index][current_raw_index]), 6)
        future_distribution = np.asarray(current_distribution @ power, dtype=float)[0]
        transition_probabilities[f"to_high_vol_stress_{horizon}d"] = round(float(future_distribution[high_vol_index]), 6)
        transition_probabilities[f"to_vol_expansion_or_high_vol_{horizon}d"] = round(float(future_distribution[expansion_index] + future_distribution[high_vol_index]), 6)

    state_usage_share_warnings = []
    total_usage = sum(int(value) for value in state_usage_counts.values()) or 1
    for label, count in state_usage_counts.items():
        if (count / total_usage) < 0.05:
            state_usage_share_warnings.append(f"HMM state `{label}` used less than 5% of the training window.")

    sector_metrics = {
        key: round(float(value), 6)
        for key, value in latest_row.items()
        if key in {"avg_pairwise_corr_21d", "first_eigenvalue_share_21d", "effective_rank_21d", "log_det_corr_21d"}
    }
    confidence = round(min(0.99, max(0.0, float(max(state_probabilities.values()) * 0.72 + 0.18))), 6)
    drivers = [
        f"{config.variant_label} top state is {top_state} with posterior probability {state_probabilities[top_state]:.2f}.",
        f"Emission-only fit points to {emission_top_state} with probability {emission_state_probabilities[emission_top_state]:.2f}.",
        f"Expected duration for the current state is {current_state_expected_duration_days:.1f} days.",
        f"5d probability of entering vol expansion or high-vol stress is {transition_probabilities['to_vol_expansion_or_high_vol_5d']:.2f}.",
    ]
    interpretation_notes = _build_interpretation_notes(
        config=config,
        top_state=top_state,
        emission_top_state=emission_top_state,
        state_probabilities=state_probabilities,
        emission_state_probabilities=emission_state_probabilities,
        persistence_lift=persistence_lift,
        transition_probabilities=transition_probabilities,
        state_feature_summaries=state_feature_summaries,
        sector_metrics=sector_metrics,
    )
    all_warnings = list(warnings) + list(repair_warnings) + state_usage_share_warnings
    return HMMBeliefRecord(
        schema_version="hmm_belief.v1",
        model_name="HMMBeliefAgent",
        model_version=str(artifact.get("model_version", config.model_version)),
        as_of=feature_record.as_of,
        is_trained=True,
        training_status="trained",
        state_probabilities=state_probabilities,
        top_state=top_state,
        transition_matrix=[[round(float(value), 6) for value in row] for row in transition_rows],
        expected_duration_days={key: round(float(value), 6) for key, value in expected_duration_days.items()},
        current_state_expected_duration_days=round(float(current_state_expected_duration_days), 6),
        persistence_probabilities=persistence_probabilities,
        transition_probabilities=transition_probabilities,
        confidence=confidence,
        warnings=all_warnings,
        drivers=drivers,
        interpretation_notes=interpretation_notes,
        state_label_mapping={str(key): value for key, value in state_mapping.items()},
        emission_state_probabilities=emission_state_probabilities,
        emission_top_state=emission_top_state,
        persistence_lift=persistence_lift,
        state_feature_summaries=state_feature_summaries,
        training_row_count=len(training_rows),
        configured_train_window=int(config.train_window),
        inference_feature_vector={key: round(float(value), 6) for key, value in latest_row.items()},
        variant_id=config.variant_id,
        variant_label=config.variant_label,
        model_converged=bool(artifact.get("model_converged", True)),
        state_usage_counts={str(key): int(value) for key, value in state_usage_counts.items()},
        sector_metrics=sector_metrics,
    )


def hmm_to_belief_record(
    hmm_record: HMMBeliefRecord,
    *,
    previous_belief: dict[str, float] | None = None,
) -> BeliefRecord:
    if not hmm_record.is_trained:
        return BeliefRecord(
            schema_version="belief.v1",
            as_of=hmm_record.as_of,
            model_version="hmm_not_trained_enough",
            beliefs={
                "STABLE_LOW_VOL_TREND": 0.0,
                "MID_VOL_CHOP": 0.0,
                "VOL_EXPANSION_TRANSITION": 0.0,
                "HIGH_VOL_RISK_OFF": 0.0,
                "PANIC_CONVEXITY_STRESS": 0.0,
                "POST_PANIC_COMPRESSION": 0.0,
            },
            belief_delta={
                "STABLE_LOW_VOL_TREND": 0.0,
                "MID_VOL_CHOP": 0.0,
                "VOL_EXPANSION_TRANSITION": 0.0,
                "HIGH_VOL_RISK_OFF": 0.0,
                "PANIC_CONVEXITY_STRESS": 0.0,
                "POST_PANIC_COMPRESSION": 0.0,
            },
            entropy=1.0,
            confidence=0.0,
            drivers=list(hmm_record.interpretation_notes[:3]),
        )
    beliefs = {
        "STABLE_LOW_VOL_TREND": round(float(hmm_record.state_probabilities.get("STABLE", 0.0)), 6),
        "MID_VOL_CHOP": 0.0,
        "VOL_EXPANSION_TRANSITION": round(float(hmm_record.state_probabilities.get("EXPANDING_VOL", 0.0)), 6),
        "HIGH_VOL_RISK_OFF": round(float(hmm_record.state_probabilities.get("HIGH_VOL", 0.0)), 6),
        "PANIC_CONVEXITY_STRESS": 0.0,
        "POST_PANIC_COMPRESSION": 0.0,
    }
    previous = previous_belief or {key: 0.0 for key in beliefs}
    total = sum(beliefs.values()) or 1.0
    beliefs = {key: round(value / total, 6) for key, value in beliefs.items()}
    belief_delta = {key: round(beliefs[key] - float(previous.get(key, 0.0)), 6) for key in beliefs}
    entropy = -sum(prob * math.log(prob) for prob in beliefs.values() if prob > 0.0)
    max_entropy = math.log(len(beliefs))
    normalized_entropy = round((entropy / max_entropy) if max_entropy > 0 else 0.0, 6)
    return BeliefRecord(
        schema_version="belief.v1",
        as_of=hmm_record.as_of,
        model_version=hmm_record.model_version,
        beliefs=beliefs,
        belief_delta=belief_delta,
        entropy=normalized_entropy,
        confidence=round(float(hmm_record.confidence), 6),
        drivers=list(hmm_record.drivers[:5]),
    )
