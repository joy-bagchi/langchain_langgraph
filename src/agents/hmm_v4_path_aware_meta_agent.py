from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from src.runtime.sklearn_runtime import configure_sklearn_runtime

configure_sklearn_runtime()

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.backtest.hmm_replay.path_aware_dataset import (
    PathAwarePrecomputedCache,
    build_path_aware_dataset,
    build_path_aware_dataset_from_precomputed,
)


STATE_ORDER = (
    "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF",
)


@dataclass(slots=True)
class HMMV5Config:
    model_name: str = "hmm_v4_path_aware_meta"
    model_type: str = "supervised_meta_learner"
    enabled: bool = True
    base_estimator: str = "gradient_boosting"
    target: str = "realized_risk_bucket"
    horizon: int = 3
    min_training_rows: int = 250
    walk_forward_train_lookback_days: int = 756
    feature_windows: list[int] = None  # type: ignore[assignment]
    geometry_stress_lookback: int = 252
    random_state: int = 42
    fallback_model: str = "hmm_v3_1_meta_blend"

    def __post_init__(self) -> None:
        if self.feature_windows is None:
            self.feature_windows = [1, 3, 5, 10, 21, 63]


def load_hmm_v4_config() -> HMMV5Config:
    path = Path(__file__).resolve().parents[2] / "agentic_vol_regime_app" / "configs" / "models" / "hmm_v4_path_aware_meta.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = payload or {}
    return HMMV5Config(
        model_name=str(payload.get("model_name", "hmm_v4_path_aware_meta")),
        model_type=str(payload.get("model_type", "supervised_meta_learner")),
        enabled=bool(payload.get("enabled", True)),
        base_estimator=str(payload.get("base_estimator", "gradient_boosting")),
        target=str(payload.get("target", "realized_risk_bucket")),
        horizon=int(payload.get("horizon", 3)),
        min_training_rows=int(payload.get("min_training_rows", 250)),
        walk_forward_train_lookback_days=int(payload.get("walk_forward_train_lookback_days", 756)),
        feature_windows=[int(item) for item in list(payload.get("feature_windows", [1, 3, 5, 10, 21, 63]))],
        geometry_stress_lookback=int(payload.get("geometry_stress_lookback", 252)),
        random_state=int(payload.get("random_state", 42)),
        fallback_model=str(payload.get("fallback_model", "hmm_v3_1_meta_blend")),
    )


def _build_estimator(config: HMMV5Config):
    if config.base_estimator == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=config.random_state,
        )
    if config.base_estimator == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
        )
    return GradientBoostingClassifier(random_state=config.random_state)


def _risk_score_to_distribution(score: float) -> dict[str, float]:
    anchors = {
        "STABLE_LOW_VOL_TREND": 0.00,
        "MID_VOL_CHOP": 0.33,
        "VOL_EXPANSION_TRANSITION": 0.67,
        "HIGH_VOL_RISK_OFF": 1.00,
    }
    temperature = 0.20
    raw = {state: float(np.exp(-abs(float(score) - anchor) / temperature)) for state, anchor in anchors.items()}
    total = sum(raw.values()) or 1.0
    return {state: float(value / total) for state, value in raw.items()}


def _importance_map(estimator, feature_columns: list[str]) -> dict[str, float]:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(getattr(estimator, "coef_"), dtype=float)
        values = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return {}
    if values.size != len(feature_columns):
        return {}
    total = float(np.sum(values)) or 1.0
    return {
        feature_columns[index]: float(values[index] / total)
        for index in range(len(feature_columns))
    }


def _group_family_importances(importances: dict[str, float], feature_families: dict[str, list[str]]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for family, columns in feature_families.items():
        grouped[family] = round(float(sum(importances.get(column, 0.0) for column in columns)), 6)
    return grouped


def _trust_weights_from_importances(importances: dict[str, float]) -> dict[str, float]:
    families = {
        "heuristic": ["heuristic_top_state_severity"],
        "hmm_v1_core": [name for name in importances if name.startswith("hmm_v1_prob_")],
        "hmm_v2_core_plus_sector_corr": [name for name in importances if name.startswith("hmm_v2_prob_")],
        "hmm_v3_core_plus_sector_geometry": [name for name in importances if name.startswith("hmm_v3_prob_")],
        "hmm_v3_1_meta_blend": [
            "hmm_v3_1_final_risk_score",
            "hmm_v3_1_final_regime_severity",
            "hmm_v3_1_geometry_stress_score",
            "hmm_v3_1_downgrade_levels",
            "hmm_v3_1_downgrade_cap_applied",
        ],
    }
    grouped = {family: sum(importances.get(column, 0.0) for column in columns) for family, columns in families.items()}
    total = sum(grouped.values())
    if total <= 0.0:
        return {family: 0.0 for family in families}
    return {family: round(float(value / total), 6) for family, value in grouped.items()}


def generate_hmm_v4_prediction(
    *,
    as_of_date: str,
    train_df: pd.DataFrame,
    fallback_prediction: dict[str, Any],
    precomputed_cache: PathAwarePrecomputedCache | None = None,
) -> dict[str, Any]:
    config = load_hmm_v4_config()
    if precomputed_cache is not None:
        dataset = build_path_aware_dataset_from_precomputed(
            cache=precomputed_cache,
            as_of_date=as_of_date,
            target_horizon=int(config.horizon),
            min_training_rows=int(config.min_training_rows),
            train_lookback_days=int(config.walk_forward_train_lookback_days),
        )
    else:
        dataset = build_path_aware_dataset(
            train_df,
            as_of_date=as_of_date,
            target_horizon=int(config.horizon),
            feature_windows=list(config.feature_windows),
            geometry_stress_lookback=int(config.geometry_stress_lookback),
            min_training_rows=int(config.min_training_rows),
        )
    warnings = list(dataset.warnings)
    if dataset.fallback_required:
        diagnostics = dict(fallback_prediction.get("model_diagnostics", {}))
        diagnostics.update(
            {
                "path_aware_estimator": config.base_estimator,
                "target_horizon": int(config.horizon),
                "training_row_count": int(len(dataset.training_frame)),
                "fallback_used": True,
                "feature_families_used": sorted(dataset.feature_families.keys()),
            }
        )
        warnings.append(
            f"HMM v4 path-aware meta model requires at least {config.min_training_rows} labeled rows; falling back to {config.fallback_model}."
        )
        result = dict(fallback_prediction)
        result["model_name"] = config.model_name
        result["warnings"] = list(dict.fromkeys(list(result.get("warnings", [])) + warnings))
        result["model_diagnostics"] = diagnostics
        result["path_features"] = {
            key: dataset.inference_row.get(key)
            for key in [
                "geometry_stress_score",
                "geometry_stress_score_delta_5d",
                "geometry_stress_score_curvature_5_10",
                "vol_geometry_gap",
                "vol_geometry_diverging",
            ]
            if key in dataset.inference_row.index
        }
        return result

    target_column = f"realized_risk_bucket_{int(config.horizon)}d"
    x_train = dataset.training_frame[dataset.feature_columns].to_numpy(dtype=float)
    y_train = (dataset.training_frame[target_column] == "HIGHER_VOL_RISK").astype(int).to_numpy(dtype=int)
    x_inference = dataset.inference_row[dataset.feature_columns].to_numpy(dtype=float).reshape(1, -1)
    if len(np.unique(y_train)) < 2:
        positive_rate = float(np.mean(y_train)) if len(y_train) else 0.0
        final_score = 1.0 if positive_rate >= 0.5 else 0.0
        state_probabilities = _risk_score_to_distribution(final_score)
        top_state = max(state_probabilities, key=state_probabilities.get)
        diagnostics = {
            "path_aware_estimator": "constant_single_class",
            "target_horizon": int(config.horizon),
            "training_row_count": int(len(dataset.training_frame)),
            "fallback_used": False,
            "feature_families_used": sorted(dataset.feature_families.keys()),
            "single_class_training": True,
            "positive_label_rate": round(positive_rate, 6),
        }
        warnings.append(
            "HMM v4 path-aware meta model training labels collapsed to one class; "
            "using constant classifier for this as-of date instead of fallback."
        )
        return {
            "model_name": config.model_name,
            "as_of_date": as_of_date,
            "target_horizon": int(config.horizon),
            "predicted_risk_bucket": "HIGHER_VOL_RISK" if final_score >= 0.5 else "LOW_RISK",
            "state_probabilities": {key: round(float(value), 6) for key, value in state_probabilities.items()},
            "top_state": top_state,
            "transition_probabilities": {
                "to_higher_vol_1d": round(float(final_score), 6),
                "to_higher_vol_2d": round(float(final_score), 6),
                "to_higher_vol_3d": round(float(final_score), 6),
            },
            "policy_output": dict(fallback_prediction.get("policy_output", {})),
            "model_diagnostics": diagnostics,
            "model_trust_weights": None,
            "path_features": {
                key: dataset.inference_row.get(key)
                for key in [
                    "geometry_stress_score",
                    "geometry_stress_score_delta_5d",
                    "geometry_stress_score_curvature_5_10",
                    "vol_geometry_gap",
                    "vol_geometry_diverging",
                ]
                if key in dataset.inference_row.index
            },
            "top_feature_importances": [],
            "warnings": warnings,
        }

    estimator = _build_estimator(config)
    estimator.fit(x_train, y_train)
    if hasattr(estimator, "predict_proba"):
        probability = float(estimator.predict_proba(x_inference)[0][1])
    else:
        score = float(estimator.decision_function(x_inference)[0])  # type: ignore[attr-defined]
        probability = float(1.0 / (1.0 + np.exp(-score)))

    final_score = min(1.0, max(0.0, probability))
    state_probabilities = _risk_score_to_distribution(final_score)
    top_state = max(state_probabilities, key=state_probabilities.get)
    feature_importances = _importance_map(estimator, dataset.feature_columns)
    grouped_importances = _group_family_importances(feature_importances, dataset.feature_families)
    trust_weights = _trust_weights_from_importances(feature_importances) if feature_importances else {}
    top_feature_importances = [
        {"feature": name, "importance": round(float(value), 6)}
        for name, value in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:10]
    ]

    diagnostics = {
        "path_aware_estimator": config.base_estimator,
        "target_horizon": int(config.horizon),
        "training_row_count": int(len(dataset.training_frame)),
        "fallback_used": False,
        "feature_families_used": sorted(dataset.feature_families.keys()),
        "grouped_feature_family_importance": grouped_importances,
    }
    return {
        "model_name": config.model_name,
        "as_of_date": as_of_date,
        "target_horizon": int(config.horizon),
        "predicted_risk_bucket": "HIGHER_VOL_RISK" if final_score >= 0.5 else "LOW_RISK",
        "state_probabilities": {key: round(float(value), 6) for key, value in state_probabilities.items()},
        "top_state": top_state,
        "transition_probabilities": {
            "to_higher_vol_1d": round(float(final_score), 6),
            "to_higher_vol_2d": round(float(final_score), 6),
            "to_higher_vol_3d": round(float(final_score), 6),
        },
        "policy_output": dict(fallback_prediction.get("policy_output", {})),
        "model_diagnostics": diagnostics,
        "model_trust_weights": trust_weights if trust_weights else None,
        "path_features": {
            key: dataset.inference_row.get(key)
            for key in [
                "geometry_stress_score",
                "geometry_stress_score_delta_5d",
                "geometry_stress_score_curvature_5_10",
                "vol_geometry_gap",
                "vol_geometry_diverging",
            ]
            if key in dataset.inference_row.index
        },
        "top_feature_importances": top_feature_importances,
        "warnings": warnings,
    }
