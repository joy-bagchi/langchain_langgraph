from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from src.runtime.sklearn_runtime import configure_sklearn_runtime

configure_sklearn_runtime()

from sklearn.preprocessing import StandardScaler

from src.agents.hmm_v4_path_aware_meta_agent import generate_hmm_v4_prediction
from src.agents.geometry_stress_agent import compute_geometry_stress
from src.backtest.hmm_replay.path_aware_dataset import PathAwarePrecomputedCache
from src.backtest.hmm_replay.replay_dataset import ensure_columns
from src.regime.meta_blend import blend_with_geometry_modifier, core_vol_risk_score

try:  # pragma: no cover
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover
    GaussianHMM = None  # type: ignore[assignment]


STATE_ORDER = (
    "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF",
)

STATE_ANCHORS = {
    "STABLE_LOW_VOL_TREND": 0.00,
    "MID_VOL_CHOP": 0.33,
    "VOL_EXPANSION_TRANSITION": 0.67,
    "HIGH_VOL_RISK_OFF": 1.00,
}

MODEL_FEATURES: dict[str, list[str]] = {
    "heuristic": [
        "vix",
        "vvix_vix_ratio",
        "realized_vol_5d",
        "realized_vol_21d",
        "term_structure_slope",
    ],
    "hmm_v1_core": [
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
    ],
    "hmm_v2_core_plus_sector_corr": [
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
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
    ],
    "hmm_v3_core_plus_sector_geometry": [
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
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
    ],
    "hmm_v3_1_meta_blend": [
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
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
    ],
    "hmm_v4_path_aware_meta": [
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
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
        "regime_target",
    ],
}
OPTIONAL_MODEL_FEATURES = {"vix9d_vix_ratio"}


def _load_meta_blend_weights() -> tuple[float, float]:
    config_path = Path(__file__).resolve().parents[3] / "agentic_vol_regime_app" / "configs" / "models" / "hmm_v3_1_meta_blend.yaml"
    if not config_path.exists():
        return (0.75, 0.25)
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        meta = dict(payload.get("meta_blend", {}))
        core_weight = float(meta.get("core_vol_weight", 0.75))
        geometry_weight = float(meta.get("geometry_weight", 0.25))
        if core_weight <= 0.0 and geometry_weight <= 0.0:
            return (0.75, 0.25)
        return (core_weight, geometry_weight)
    except Exception:
        return (0.75, 0.25)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


@dataclass(slots=True)
class ReplayContext:
    run_id: str
    as_of_date: str
    mode: str = "historical_replay"
    allow_live_data: bool = False
    allow_production_artifact_write: bool = False


def create_replay_context(as_of_date: str) -> ReplayContext:
    return ReplayContext(run_id=str(uuid4()), as_of_date=as_of_date)


def _risk_score(row: pd.Series) -> float:
    return (
        float(row.get("vix", 0.0))
        + float(row.get("realized_vol_21d", 0.0))
        + (float(row.get("drawdown_21d", 0.0)) * 35.0)
        + float(row.get("vvix_vix_ratio", 0.0))
        - (float(row.get("trend_persistence_21d", 0.0)) * 8.0)
        - float(row.get("term_structure_slope", 0.0))
    )


def _map_policy(top_state: str, spy_close: float) -> dict[str, Any]:
    if top_state == "STABLE_LOW_VOL_TREND":
        posture = "NO_OVERWRITE"
        dte = 3
        offset = 0.012
    elif top_state == "MID_VOL_CHOP":
        posture = "LIGHT_OVERWRITE"
        dte = 2
        offset = 0.009
    elif top_state == "VOL_EXPANSION_TRANSITION":
        posture = "MEDIUM_OVERWRITE"
        dte = 1
        offset = 0.006
    else:
        posture = "AGGRESSIVE_OVERWRITE"
        dte = 1
        offset = 0.004
    strike = round(float(spy_close) * (1.0 + offset), 2)
    return {
        "overwrite_posture": posture,
        "suggested_dte": int(dte),
        "suggested_delta": 0.25 if dte > 1 else 0.2,
        "suggested_strike": strike,
    }


def _validate_training(train_df: pd.DataFrame, *, as_of_date: date, min_train_rows: int) -> None:
    if train_df.empty:
        raise RuntimeError("Replay training slice is empty.")
    if train_df["date"].max() > as_of_date:
        raise RuntimeError("Replay no-lookahead violation: training slice includes rows after as_of_date.")
    if len(train_df) < int(min_train_rows):
        raise RuntimeError(f"Replay training slice has {len(train_df)} rows; requires at least {min_train_rows}.")


def _heuristic_prediction(train_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    row = train_df.iloc[-1]
    risk = _risk_score(row)
    chop = abs(float(row.get("spy_return_1d", 0.0))) * 200.0 + float(row.get("vvix_vix_ratio", 0.0))
    raw = np.asarray(
        [
            max(0.0, 40.0 - risk),
            max(0.0, 15.0 + chop - abs(risk - 25.0)),
            max(0.0, risk),
            max(0.0, risk - 20.0),
        ],
        dtype=float,
    )
    probs = raw / (float(np.sum(raw)) or 1.0)
    state_probs = {state: round(float(prob), 6) for state, prob in zip(STATE_ORDER, probs)}
    top_state = max(state_probs, key=state_probs.get)
    transition_probs = {
        "to_higher_vol_1d": round(float(state_probs["VOL_EXPANSION_TRANSITION"] + state_probs["HIGH_VOL_RISK_OFF"]), 6),
        "to_higher_vol_2d": round(float(min(1.0, state_probs["VOL_EXPANSION_TRANSITION"] + state_probs["HIGH_VOL_RISK_OFF"] + 0.05)), 6),
        "to_higher_vol_3d": round(float(min(1.0, state_probs["VOL_EXPANSION_TRANSITION"] + state_probs["HIGH_VOL_RISK_OFF"] + 0.1)), 6),
    }
    diagnostics = {
        "converged": True,
        "state_usage_counts": {},
        "state_means": {},
        "top_state": top_state,
    }
    return state_probs, transition_probs, diagnostics


def _hmm_prediction(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    n_components: int,
    random_state: int,
    covariance_type: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], list[list[float]], dict[str, float]]:
    if GaussianHMM is None:
        raise RuntimeError("hmmlearn is required for HMM replay predictions.")
    feature_cols = list(MODEL_FEATURES[model_name])
    dropped_optional_features: list[str] = []
    for feature_name in list(feature_cols):
        if feature_name not in OPTIONAL_MODEL_FEATURES:
            continue
        if feature_name not in train_df.columns:
            feature_cols.remove(feature_name)
            dropped_optional_features.append(feature_name)
            continue
        coverage = float(pd.to_numeric(train_df[feature_name], errors="coerce").notna().mean())
        if coverage < 0.80:
            feature_cols.remove(feature_name)
            dropped_optional_features.append(feature_name)
    ensure_columns(train_df, feature_cols, context=f"{model_name} replay")
    clean = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    if clean.empty:
        raise RuntimeError(f"{model_name} has no non-null rows after feature filtering.")
    matrix = clean[feature_cols].to_numpy(dtype=float)
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        raise RuntimeError(f"{model_name} replay features contain NaN/inf after preprocessing.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = GaussianHMM(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        n_iter=500,
        random_state=int(random_state),
    )
    model.fit(scaled)
    inference_window = scaled[-min(63, len(scaled)) :]
    posterior = np.asarray(model.predict_proba(inference_window)[-1], dtype=float)
    risk_scores = []
    means_unscaled = scaler.inverse_transform(model.means_)
    for state_idx in range(model.n_components):
        vec = means_unscaled[state_idx]
        row_like = pd.Series({name: vec[i] for i, name in enumerate(feature_cols)})
        risk_scores.append((state_idx, _risk_score(row_like)))
    ordered = [idx for idx, _ in sorted(risk_scores, key=lambda item: item[1])]
    mapped = {ordered[min(i, len(ordered) - 1)]: STATE_ORDER[i] for i in range(min(len(STATE_ORDER), len(ordered)))}
    state_probs = {state: 0.0 for state in STATE_ORDER}
    for idx, prob in enumerate(posterior):
        state_probs[mapped.get(idx, STATE_ORDER[-1])] += float(prob)
    total = float(sum(state_probs.values())) or 1.0
    state_probs = {state: round(value / total, 6) for state, value in state_probs.items()}
    transition = np.asarray(model.transmat_, dtype=float)
    current_idx = int(np.argmax(posterior))
    p1 = transition[current_idx]
    p2 = p1 @ transition
    p3 = p2 @ transition
    high_indices = [idx for idx, state in mapped.items() if state in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"}]
    transition_probs = {
        "to_higher_vol_1d": round(float(sum(p1[idx] for idx in high_indices)), 6),
        "to_higher_vol_2d": round(float(sum(p2[idx] for idx in high_indices)), 6),
        "to_higher_vol_3d": round(float(sum(p3[idx] for idx in high_indices)), 6),
    }
    counts = np.bincount(np.argmax(model.predict_proba(scaled), axis=1), minlength=model.n_components)
    state_usage_counts = {mapped.get(i, str(i)): int(counts[i]) for i in range(model.n_components)}
    state_means = {
        mapped.get(i, str(i)): {
            name: round(float(means_unscaled[i, j]), 6)
            for j, name in enumerate(feature_cols)
        }
        for i in range(model.n_components)
    }
    expected_duration = {}
    for idx in range(model.n_components):
        persistence = float(transition[idx, idx])
        duration = 999.0 if persistence >= 0.999 else (1.0 / max(1e-6, 1.0 - persistence))
        expected_duration[mapped.get(idx, str(idx))] = round(float(duration), 6)
    diagnostics = {
        "converged": bool(getattr(getattr(model, "monitor_", None), "converged", True)),
        "state_usage_counts": state_usage_counts,
        "state_means": state_means,
        "dropped_optional_features": dropped_optional_features,
    }
    return (
        state_probs,
        transition_probs,
        diagnostics,
        [[round(float(item), 6) for item in row] for row in transition.tolist()],
        expected_duration,
    )


def _risk_score_to_distribution(score: float) -> dict[str, float]:
    values: dict[str, float] = {}
    temperature = 0.20
    for state in STATE_ORDER:
        anchor = float(STATE_ANCHORS[state])
        values[state] = float(np.exp(-abs(float(score) - anchor) / temperature))
    total = sum(values.values()) or 1.0
    return {state: round(values[state] / total, 6) for state in STATE_ORDER}


def _meta_blend_prediction(
    train_df: pd.DataFrame,
    *,
    n_components: int,
    random_state: int,
    covariance_type: str,
    core_vol_weight: float | None = None,
    geometry_weight: float | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], list[list[float]], dict[str, float], list[str]]:
    if core_vol_weight is None or geometry_weight is None:
        default_core, default_geo = _load_meta_blend_weights()
        core_vol_weight = float(default_core if core_vol_weight is None else core_vol_weight)
        geometry_weight = float(default_geo if geometry_weight is None else geometry_weight)
    core_probs, core_transition_probs, core_diagnostics, core_transition_matrix, core_expected_duration = _hmm_prediction(
        train_df,
        model_name="hmm_v1_core",
        n_components=n_components,
        random_state=random_state,
        covariance_type=covariance_type,
    )
    core_state = max(core_probs, key=core_probs.get)
    core_confidence = float(core_probs.get(core_state, 0.0))
    core_score = core_vol_risk_score(core_probs)

    geometry_output = compute_geometry_stress(train_df)
    blend = blend_with_geometry_modifier(
        core_vol_state=core_state,
        core_vol_confidence=core_confidence,
        core_vol_risk_score_value=core_score,
        geometry_stress_score=float(geometry_output.geometry_stress_score),
        core_vol_weight=core_vol_weight,
        geometry_weight=geometry_weight,
    )

    final_probs = _risk_score_to_distribution(blend.final_risk_score)
    final_transition_probs = {
        key: round(
            float((0.75 * float(core_transition_probs.get(key, 0.0))) + (0.25 * blend.final_risk_score)),
            6,
        )
        for key in ("to_higher_vol_1d", "to_higher_vol_2d", "to_higher_vol_3d")
    }
    diagnostics = {
        **core_diagnostics,
        "core_model": "hmm_v1_core",
        "core_vol_state": core_state,
        "core_vol_confidence": round(core_confidence, 6),
        "core_vol_risk_score": round(float(core_score), 6),
        "geometry_stress_score": round(float(geometry_output.geometry_stress_score), 6),
        "geometry_confirmation_level": geometry_output.geometry_confirmation_level,
        "geometry_components": dict(geometry_output.geometry_components),
        "lookback_used": int(geometry_output.lookback_used),
        "final_risk_score": round(float(blend.final_risk_score), 6),
        "final_regime": blend.final_regime,
        "confidence_adjustment": blend.confidence_adjustment,
        "downgrade_levels": int(blend.downgrade_levels),
        "downgrade_cap_applied": bool(blend.downgrade_cap_applied),
    }
    warnings = list(geometry_output.warnings)
    return (
        final_probs,
        final_transition_probs,
        diagnostics,
        core_transition_matrix,
        core_expected_duration,
        warnings + list(geometry_output.geometry_rationale) + list(blend.rationale),
    )


def generate_prediction_record(
    *,
    context: ReplayContext,
    model_name: str,
    train_df: pd.DataFrame,
    min_train_rows: int,
    n_components: int,
    random_state: int,
    covariance_type: str,
    precomputed_path_aware_cache: PathAwarePrecomputedCache | None = None,
) -> dict[str, Any]:
    as_of = pd.to_datetime(context.as_of_date).date()
    if model_name != "hmm_v4_path_aware_meta":
        _validate_training(train_df, as_of_date=as_of, min_train_rows=min_train_rows)
    elif train_df.empty or train_df["date"].max() > as_of:
        _validate_training(train_df, as_of_date=as_of, min_train_rows=1)
    if model_name not in MODEL_FEATURES:
        raise RuntimeError(f"Unsupported replay model: {model_name}")

    warnings: list[str] = []
    if model_name == "heuristic":
        state_probs, transition_probs, diagnostics = _heuristic_prediction(train_df)
        transition_matrix: list[list[float]] = []
        expected_duration: dict[str, float] = {}
    elif model_name == "hmm_v4_path_aware_meta":
        fallback_record = generate_prediction_record(
            context=context,
            model_name="hmm_v3_1_meta_blend",
            train_df=train_df,
            min_train_rows=min_train_rows,
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
        )
        v5_record = generate_hmm_v4_prediction(
            as_of_date=context.as_of_date,
            train_df=train_df,
            fallback_prediction=fallback_record,
            precomputed_cache=precomputed_path_aware_cache,
        )
        record = {
            "run_id": context.run_id,
            "as_of_date": context.as_of_date,
            "model_name": model_name,
            "top_state": v5_record["top_state"],
            "state_probabilities": dict(v5_record["state_probabilities"]),
            "transition_probabilities": dict(v5_record["transition_probabilities"]),
            "policy_output": dict(v5_record["policy_output"]),
            "feature_snapshot": {
                column: (_safe_float(train_df.iloc[-1].get(column)) if column in train_df.columns else None)
                for column in MODEL_FEATURES["hmm_v3_1_meta_blend"]
            },
            "model_diagnostics": {
                **dict(v5_record.get("model_diagnostics", {})),
                "predicted_risk_bucket": v5_record.get("predicted_risk_bucket"),
                "model_trust_weights": v5_record.get("model_trust_weights"),
                "path_features": v5_record.get("path_features"),
                "top_feature_importances": v5_record.get("top_feature_importances"),
            },
            "transition_matrix": list(fallback_record.get("transition_matrix", [])),
            "expected_duration_days": dict(fallback_record.get("expected_duration_days", {})),
            "warnings": list(v5_record.get("warnings", [])),
        }
        return record
    elif model_name == "hmm_v3_1_meta_blend":
        (
            state_probs,
            transition_probs,
            diagnostics,
            transition_matrix,
            expected_duration,
            warnings,
        ) = _meta_blend_prediction(
            train_df,
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
        )
    else:
        state_probs, transition_probs, diagnostics, transition_matrix, expected_duration = _hmm_prediction(
            train_df,
            model_name=model_name,
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
        )

    top_state = max(state_probs, key=state_probs.get)
    as_of_row = train_df.iloc[-1]
    policy_output = _map_policy(top_state, float(as_of_row["spy_close"]))
    feature_snapshot = {
        key: (float(as_of_row[key]) if key in as_of_row and pd.notna(as_of_row[key]) else None)
        for key in MODEL_FEATURES[model_name]
    }
    return {
        "run_id": context.run_id,
        "as_of_date": context.as_of_date,
        "model_name": model_name,
        "top_state": top_state,
        "state_probabilities": state_probs,
        "transition_matrix": transition_matrix,
        "expected_duration_days": expected_duration,
        "transition_probabilities": transition_probs,
        "policy_output": policy_output,
        "feature_snapshot": feature_snapshot,
        "warnings": warnings,
        "model_diagnostics": {
            **diagnostics,
            "training_row_count": int(len(train_df)),
            "training_end_date": str(train_df["date"].max()),
        },
    }
