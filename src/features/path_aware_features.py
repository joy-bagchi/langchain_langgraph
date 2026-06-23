from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.agents.geometry_stress_agent import compute_geometry_stress


GEOMETRY_LEVEL_COLUMNS = (
    "avg_pairwise_corr_21d",
    "first_eigenvalue_share_21d",
    "effective_rank_21d",
    "log_det_corr_21d",
)

GEOMETRY_COMPONENT_COLUMNS = (
    "avg_corr_stress",
    "eigen_stress",
    "effective_rank_stress",
    "log_det_stress",
)

VOL_PATH_COLUMNS = (
    "vix",
    "vvix",
    "vvix_vix_ratio",
    "vix_vix3m_ratio",
    "vix9d_vix_ratio",
    "realized_vol_21d",
    "drawdown_21d",
    "trend_persistence_21d",
)
OPTIONAL_VOL_PATH_COLUMNS = {"vix9d_vix_ratio"}

STATE_ORDER = (
    "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF",
)

STATE_SEVERITY = {
    "STABLE_LOW_VOL_TREND": 0.0,
    "MID_VOL_CHOP": 0.33,
    "VOL_EXPANSION_TRANSITION": 0.67,
    "HIGH_VOL_RISK_OFF": 1.0,
}


@dataclass(slots=True)
class PathAwareFeatureBundle:
    features: pd.DataFrame
    warnings: list[str]
    feature_families: dict[str, list[str]]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _state_probs_from_score(score: float) -> dict[str, float]:
    temperature = 0.20
    raw = {
        state: float(np.exp(-abs(float(score) - float(anchor)) / temperature))
        for state, anchor in STATE_SEVERITY.items()
    }
    total = sum(raw.values()) or 1.0
    return {state: float(value / total) for state, value in raw.items()}


def _add_delta_and_slope(
    frame: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    *,
    slope_only: set[str],
) -> tuple[dict[str, pd.Series], list[str]]:
    new_features: dict[str, pd.Series] = {}
    created: list[str] = []
    for column in columns:
        for window in windows:
            delta_col = f"{column}_delta_{window}d"
            delta_series = frame[column] - frame[column].shift(window)
            new_features[delta_col] = delta_series
            created.append(delta_col)
            if column in slope_only and window in {3, 5, 10, 21}:
                slope_col = f"{column}_slope_{window}d"
                new_features[slope_col] = delta_series / float(window)
                created.append(slope_col)
    return new_features, created


def _curvature(current: pd.Series, *, short_window: int, long_window: int) -> pd.Series:
    recent = (current - current.shift(short_window)) / float(short_window)
    prior = (current.shift(short_window) - current.shift(long_window)) / float(long_window - short_window)
    return recent - prior


def _rolling_count(series: pd.Series, predicate) -> pd.Series:
    indicator = series.apply(lambda value: 1.0 if predicate(float(value)) else 0.0)
    return indicator


def build_path_aware_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_windows: list[int],
    geometry_stress_lookback: int,
) -> PathAwareFeatureBundle:
    required = {
        "date",
        *GEOMETRY_LEVEL_COLUMNS,
        *(column for column in VOL_PATH_COLUMNS if column not in OPTIONAL_VOL_PATH_COLUMNS),
    }
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise RuntimeError(f"Path-aware feature frame missing required columns: {', '.join(missing)}")

    working = frame.copy().sort_values("date").reset_index(drop=True)
    warnings: list[str] = []
    active_vol_path_columns: list[str] = []
    for column in VOL_PATH_COLUMNS:
        if column not in working.columns:
            if column in OPTIONAL_VOL_PATH_COLUMNS:
                warnings.append(f"Optional path column '{column}' is unavailable and will be excluded.")
                continue
            raise RuntimeError(f"Path-aware feature frame missing required column: {column}")
        if column in OPTIONAL_VOL_PATH_COLUMNS:
            coverage = float(pd.to_numeric(working[column], errors="coerce").notna().mean())
            if coverage < 0.80:
                warnings.append(
                    f"Optional path column '{column}' has insufficient coverage ({coverage:.2%}) and will be excluded."
                )
                continue
        active_vol_path_columns.append(column)
    geometry_rows: list[dict[str, Any]] = []
    for index in range(len(working)):
        scoped = working.iloc[: index + 1].copy()
        output = compute_geometry_stress(
            scoped,
            preferred_lookback=int(geometry_stress_lookback),
            fallback_lookback=max(63, int(min(126, geometry_stress_lookback))),
            min_lookback=63,
        )
        if output.warnings:
            warnings.extend(str(item) for item in output.warnings)
        geometry_rows.append(
            {
                "geometry_stress_score": float(output.geometry_stress_score),
                "avg_corr_stress": float(output.geometry_components["avg_corr_stress"]),
                "eigen_stress": float(output.geometry_components["eigen_stress"]),
                "effective_rank_stress": float(output.geometry_components["effective_rank_stress"]),
                "log_det_stress": float(output.geometry_components["log_det_stress"]),
                "geometry_confirmation_level": str(output.geometry_confirmation_level),
                "geometry_lookback_used": float(output.lookback_used),
            }
        )
    geometry_df = pd.DataFrame(geometry_rows)
    working = pd.concat([working.reset_index(drop=True), geometry_df], axis=1)

    level_family = [
        "geometry_stress_score",
        *GEOMETRY_COMPONENT_COLUMNS,
        *GEOMETRY_LEVEL_COLUMNS,
    ]
    delta_columns = [
        "geometry_stress_score",
        *GEOMETRY_COMPONENT_COLUMNS,
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
        *active_vol_path_columns,
    ]
    slope_base = {
        "geometry_stress_score",
        "avg_corr_stress",
        "eigen_stress",
        "effective_rank_stress",
        "log_det_stress",
    }
    new_features: dict[str, pd.Series] = {}
    delta_features, delta_family = _add_delta_and_slope(working, delta_columns, feature_windows, slope_only=slope_base)
    new_features.update(delta_features)

    curvature_specs = [
        ("geometry_stress_score", 5, 10),
        ("geometry_stress_score", 10, 21),
        ("eigen_stress", 5, 10),
        ("effective_rank_stress", 5, 10),
        ("log_det_stress", 5, 10),
    ]
    curvature_family: list[str] = []
    for name, short_window, long_window in curvature_specs:
        column_name = f"{name}_curvature_{short_window}_{long_window}"
        new_features[column_name] = _curvature(working[name], short_window=short_window, long_window=long_window)
        curvature_family.append(column_name)

    persistence_family: list[str] = []
    above_055 = _rolling_count(working["geometry_stress_score"], lambda value: value > 0.55)
    above_070 = _rolling_count(working["geometry_stress_score"], lambda value: value > 0.70)
    below_030 = _rolling_count(working["geometry_stress_score"], lambda value: value < 0.30)
    for window in [5, 10, 21, 63]:
        new_features[f"geometry_days_above_0_55_{window}d"] = above_055.rolling(window, min_periods=1).sum()
        new_features[f"geometry_days_above_0_70_{window}d"] = above_070.rolling(window, min_periods=1).sum()
        new_features[f"geometry_days_below_0_30_{window}d"] = below_030.rolling(window, min_periods=1).sum()
        new_features[f"geometry_mean_{window}d"] = working["geometry_stress_score"].rolling(window, min_periods=1).mean()
        new_features[f"geometry_max_{window}d"] = working["geometry_stress_score"].rolling(window, min_periods=1).max()
        new_features[f"geometry_min_{window}d"] = working["geometry_stress_score"].rolling(window, min_periods=1).min()
        new_features[f"geometry_std_{window}d"] = (
            working["geometry_stress_score"].rolling(window, min_periods=2).std().fillna(0.0)
        )
        persistence_family.extend(
            [
                f"geometry_days_above_0_55_{window}d",
                f"geometry_days_above_0_70_{window}d",
                f"geometry_days_below_0_30_{window}d",
                f"geometry_mean_{window}d",
                f"geometry_max_{window}d",
                f"geometry_min_{window}d",
                f"geometry_std_{window}d",
            ]
        )

    core_vol_risk_score = (
        (working["vix"].rank(pct=True).fillna(0.5) * 0.35)
        + (working["realized_vol_21d"].rank(pct=True).fillna(0.5) * 0.25)
        + (working["vvix_vix_ratio"].rank(pct=True).fillna(0.5) * 0.20)
        + ((1.0 - working["term_structure_slope"].rank(pct=True).fillna(0.5)) * 0.20)
    ).clip(0.0, 1.0)
    vol_geometry_gap = core_vol_risk_score - working["geometry_stress_score"]
    new_features["core_vol_risk_score"] = core_vol_risk_score
    new_features["vol_geometry_gap"] = vol_geometry_gap
    new_features["vol_geometry_gap_abs"] = vol_geometry_gap.abs()
    new_features["vol_geometry_confirming"] = (
        (core_vol_risk_score > 0.55) & (working["geometry_stress_score"] > 0.55)
    ).astype(int)
    new_features["vol_geometry_diverging"] = (
        ((core_vol_risk_score > 0.60) & (working["geometry_stress_score"] < 0.35))
        | ((working["geometry_stress_score"] > 0.60) & (core_vol_risk_score < 0.35))
    ).astype(int)
    divergence_family = [
        "core_vol_risk_score",
        "vol_geometry_gap",
        "vol_geometry_gap_abs",
        "vol_geometry_confirming",
        "vol_geometry_diverging",
    ]
    for window in [5, 10]:
        column = f"vol_geometry_gap_delta_{window}d"
        new_features[column] = vol_geometry_gap - vol_geometry_gap.shift(window)
        divergence_family.append(column)

    features_df = pd.DataFrame(new_features, index=working.index)
    working = pd.concat([working, features_df], axis=1).copy()

    severity_distributions = [
        _state_probs_from_score(float(score))
        for score in working["core_vol_risk_score"].fillna(0.5).tolist()
    ]
    ensemble_rows: list[dict[str, Any]] = []
    for index, row in working.iterrows():
        heuristic_severity = float(_clip01(
            0.35 * float(row.get("vix", 0.0) / max(1.0, float(working["vix"].max() or 1.0)))
            + 0.25 * float(row.get("realized_vol_21d", 0.0) / max(1.0, float(working["realized_vol_21d"].max() or 1.0)))
            + 0.20 * float(row.get("geometry_stress_score", 0.5))
            + 0.20 * float(row.get("core_vol_risk_score", 0.5))
        ))
        v1_probs = severity_distributions[index]
        v2_score = _clip01(float(row.get("core_vol_risk_score", 0.5)) + 0.08 * float(row.get("eigen_stress", 0.5)))
        v3_score = _clip01(v2_score + 0.08 * float(row.get("effective_rank_stress", 0.5)))
        v3_1_score = _clip01((0.75 * float(row.get("core_vol_risk_score", 0.5))) + (0.25 * float(row.get("geometry_stress_score", 0.5))))
        v2_probs = _state_probs_from_score(v2_score)
        v3_probs = _state_probs_from_score(v3_score)
        v3_1_probs = _state_probs_from_score(v3_1_score)
        sev_values = [
            heuristic_severity,
            float(max(v1_probs, key=v1_probs.get) and max(STATE_SEVERITY[state] * prob for state, prob in v1_probs.items())),
            v2_score,
            v3_score,
            v3_1_score,
        ]
        higher_vol_count = 0
        low_risk_count = 0
        for score in (heuristic_severity, float(row.get("core_vol_risk_score", 0.5)), v2_score, v3_score, v3_1_score):
            if score >= 0.50:
                higher_vol_count += 1
            else:
                low_risk_count += 1
        ensemble_rows.append(
            {
                "heuristic_top_state_severity": heuristic_severity,
                **{f"hmm_v1_prob_{state.lower()}": float(v1_probs[state]) for state in STATE_ORDER},
                **{f"hmm_v2_prob_{state.lower()}": float(v2_probs[state]) for state in STATE_ORDER},
                **{f"hmm_v3_prob_{state.lower()}": float(v3_probs[state]) for state in STATE_ORDER},
                "hmm_v3_1_final_risk_score": v3_1_score,
                "hmm_v3_1_final_regime_severity": max(STATE_SEVERITY[state] * prob for state, prob in v3_1_probs.items()),
                "hmm_v3_1_geometry_stress_score": float(row.get("geometry_stress_score", 0.5)),
                "hmm_v3_1_downgrade_levels": int(
                    max(0, round((float(row.get("core_vol_risk_score", 0.5)) - v3_1_score) / 0.33))
                ),
                "hmm_v3_1_downgrade_cap_applied": int(
                    float(row.get("core_vol_risk_score", 0.5)) >= 0.67 and v3_1_score < float(row.get("core_vol_risk_score", 0.5)) - 0.33
                ),
                "v1_v3_severity_gap": float(row.get("core_vol_risk_score", 0.5)) - v3_score,
                "v1_v3_1_severity_gap": float(row.get("core_vol_risk_score", 0.5)) - v3_1_score,
                "v1_v2_severity_gap": float(row.get("core_vol_risk_score", 0.5)) - v2_score,
                "number_of_models_predicting_higher_vol": higher_vol_count,
                "number_of_models_predicting_low_risk": low_risk_count,
                "max_model_severity": max(sev_values),
                "min_model_severity": min(sev_values),
                "model_severity_dispersion": float(np.std(np.asarray(sev_values, dtype=float))),
            }
        )
    ensemble_df = pd.DataFrame(ensemble_rows)
    working = pd.concat([working, ensemble_df], axis=1)
    ensemble_family = list(ensemble_df.columns)

    for column in working.columns:
        if column == "date" or column == "geometry_confirmation_level":
            continue
        working[column] = pd.to_numeric(working[column], errors="coerce")

    families = {
        "levels": level_family,
        "deltas_slopes": delta_family,
        "curvature": curvature_family,
        "persistence": persistence_family,
        "divergence": divergence_family,
        "ensemble": ensemble_family,
    }
    return PathAwareFeatureBundle(features=working, warnings=sorted(set(warnings)), feature_families=families)
