from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


GEOMETRY_COMPONENT_COLUMNS = (
    "avg_pairwise_corr_21d",
    "first_eigenvalue_share_21d",
    "effective_rank_21d",
    "log_det_corr_21d",
)


@dataclass(slots=True, frozen=True)
class GeometryStressOutput:
    geometry_stress_score: float
    geometry_confirmation_level: str
    geometry_components: dict[str, float]
    lookback_used: int
    warnings: list[str]
    geometry_rationale: list[str]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _percentile_rank(values: Iterable[float], current: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.5
    rank = float(np.mean(arr <= float(current)))
    return _clip01(rank)


def _confirmation_level(score: float) -> str:
    if score < 0.30:
        return "geometry_not_confirming"
    if score < 0.55:
        return "geometry_mild_confirmation"
    if score < 0.75:
        return "geometry_confirming"
    return "geometry_strong_confirmation"


def compute_geometry_stress(
    history_df: pd.DataFrame,
    *,
    preferred_lookback: int = 252,
    fallback_lookback: int = 126,
    min_lookback: int = 63,
) -> GeometryStressOutput:
    clean = history_df.copy()
    for column in GEOMETRY_COMPONENT_COLUMNS:
        if column not in clean.columns:
            return GeometryStressOutput(
                geometry_stress_score=0.50,
                geometry_confirmation_level="geometry_mild_confirmation",
                geometry_components={
                    "avg_corr_stress": 0.5,
                    "eigen_stress": 0.5,
                    "effective_rank_stress": 0.5,
                    "log_det_stress": 0.5,
                },
                lookback_used=0,
                warnings=[f"Missing geometry column: {column}"],
                geometry_rationale=["Geometry stress set to neutral because one or more geometry features were missing."],
            )
    clean = clean.dropna(subset=list(GEOMETRY_COMPONENT_COLUMNS)).reset_index(drop=True)
    if clean.empty:
        return GeometryStressOutput(
            geometry_stress_score=0.50,
            geometry_confirmation_level="geometry_mild_confirmation",
            geometry_components={
                "avg_corr_stress": 0.5,
                "eigen_stress": 0.5,
                "effective_rank_stress": 0.5,
                "log_det_stress": 0.5,
            },
            lookback_used=0,
            warnings=["No usable geometry history after filtering NaN values."],
            geometry_rationale=["Geometry stress set to neutral because geometry history was empty."],
        )
    if len(clean) >= preferred_lookback:
        lookback = preferred_lookback
    elif len(clean) >= fallback_lookback:
        lookback = fallback_lookback
    elif len(clean) >= min_lookback:
        lookback = min_lookback
    else:
        return GeometryStressOutput(
            geometry_stress_score=0.50,
            geometry_confirmation_level="geometry_mild_confirmation",
            geometry_components={
                "avg_corr_stress": 0.5,
                "eigen_stress": 0.5,
                "effective_rank_stress": 0.5,
                "log_det_stress": 0.5,
            },
            lookback_used=int(len(clean)),
            warnings=[
                "Insufficient geometry history for stress scoring; using neutral geometry stress 0.50."
            ],
            geometry_rationale=["Geometry stress set to neutral due to insufficient lookback history."],
        )
    window = clean.tail(lookback).reset_index(drop=True)
    latest = window.iloc[-1]

    avg_corr_stress = _percentile_rank(window["avg_pairwise_corr_21d"].tolist(), float(latest["avg_pairwise_corr_21d"]))
    eigen_stress = _percentile_rank(window["first_eigenvalue_share_21d"].tolist(), float(latest["first_eigenvalue_share_21d"]))
    effective_rank_stress = _percentile_rank((-window["effective_rank_21d"]).tolist(), float(-latest["effective_rank_21d"]))
    log_det_stress = _percentile_rank((-window["log_det_corr_21d"]).tolist(), float(-latest["log_det_corr_21d"]))

    score = _clip01(
        (0.30 * avg_corr_stress)
        + (0.30 * eigen_stress)
        + (0.25 * effective_rank_stress)
        + (0.15 * log_det_stress)
    )
    level = _confirmation_level(score)
    rationale = [
        f"Geometry stress uses lookback={lookback}.",
        f"avg_corr_stress={avg_corr_stress:.2f}, eigen_stress={eigen_stress:.2f}, "
        f"effective_rank_stress={effective_rank_stress:.2f}, log_det_stress={log_det_stress:.2f}.",
    ]
    return GeometryStressOutput(
        geometry_stress_score=score,
        geometry_confirmation_level=level,
        geometry_components={
            "avg_corr_stress": round(avg_corr_stress, 6),
            "eigen_stress": round(eigen_stress, 6),
            "effective_rank_stress": round(effective_rank_stress, 6),
            "log_det_stress": round(log_det_stress, 6),
        },
        lookback_used=lookback,
        warnings=[],
        geometry_rationale=rationale,
    )
