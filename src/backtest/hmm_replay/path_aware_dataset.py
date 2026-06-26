from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.features.path_aware_features import PathAwareFeatureBundle, build_path_aware_feature_frame


HIGHER_VOL_STATES = {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"}


@dataclass(slots=True)
class PathAwareDatasetBundle:
    training_frame: pd.DataFrame
    inference_row: pd.Series
    feature_columns: list[str]
    feature_families: dict[str, list[str]]
    warnings: list[str]
    fallback_required: bool


@dataclass(slots=True)
class PathAwarePrecomputedCache:
    enriched_frame: pd.DataFrame
    feature_families: dict[str, list[str]]
    warnings: list[str]
    date_to_last_index: dict[str, int]


def _risk_bucket_from_state(value: str | None) -> str | None:
    if value is None:
        return None
    return "HIGHER_VOL_RISK" if str(value) in HIGHER_VOL_STATES else "LOW_RISK"


def _build_targets(frame: pd.DataFrame, *, horizons: list[int]) -> pd.DataFrame:
    working = frame.copy()
    for horizon in horizons:
        working[f"realized_state_{horizon}d"] = working["regime_target"].shift(-horizon)
        working[f"realized_risk_bucket_{horizon}d"] = working[f"realized_state_{horizon}d"].map(_risk_bucket_from_state)
        vix_future = working["vix"].shift(-horizon)
        rv_future = working["realized_vol_21d"].shift(-horizon)
        working[f"vix_spike_{horizon}d"] = (
            ((vix_future - working["vix"]) / working["vix"].replace(0.0, pd.NA)).fillna(0.0) >= 0.10
        ).astype(int)
        working[f"rv_expanded_{horizon}d"] = (rv_future > working["realized_vol_21d"]).astype(int)
        working[f"higher_vol_transition_{horizon}d"] = (
            working[f"realized_risk_bucket_{horizon}d"] == "HIGHER_VOL_RISK"
        ).astype(int)
    return working


def build_path_aware_precomputed_cache(
    frame: pd.DataFrame,
    *,
    feature_windows: list[int],
    geometry_stress_lookback: int,
) -> PathAwarePrecomputedCache:
    if frame.empty:
        raise RuntimeError("Path-aware precompute received an empty frame.")
    working = frame.copy().sort_values("date").reset_index(drop=True)
    if "regime_target" not in working.columns:
        raise RuntimeError("Path-aware precompute requires a 'regime_target' column.")
    feature_bundle: PathAwareFeatureBundle = build_path_aware_feature_frame(
        working,
        feature_windows=feature_windows,
        geometry_stress_lookback=geometry_stress_lookback,
    )
    enriched = _build_targets(feature_bundle.features, horizons=[1, 2, 3, 5])
    date_to_last_index: dict[str, int] = {}
    for index, day in enumerate(enriched["date"].tolist()):
        date_to_last_index[str(day)] = int(index)
    return PathAwarePrecomputedCache(
        enriched_frame=enriched,
        feature_families=dict(feature_bundle.feature_families),
        warnings=list(feature_bundle.warnings),
        date_to_last_index=date_to_last_index,
    )


def build_path_aware_dataset(
    train_df: pd.DataFrame,
    *,
    as_of_date: str,
    target_horizon: int,
    feature_windows: list[int],
    geometry_stress_lookback: int,
    min_training_rows: int,
) -> PathAwareDatasetBundle:
    if train_df.empty:
        raise RuntimeError("Path-aware dataset builder received an empty training frame.")
    working = train_df.copy().sort_values("date").reset_index(drop=True)
    if "regime_target" not in working.columns:
        raise RuntimeError("Path-aware dataset requires a 'regime_target' column in the replay feature store.")

    feature_bundle: PathAwareFeatureBundle = build_path_aware_feature_frame(
        working,
        feature_windows=feature_windows,
        geometry_stress_lookback=geometry_stress_lookback,
    )
    enriched = _build_targets(feature_bundle.features, horizons=[1, 2, 3, 5])
    inference_mask = enriched["date"] == pd.to_datetime(as_of_date).date()
    if not inference_mask.any():
        raise RuntimeError(f"Path-aware dataset could not find inference row for as_of_date={as_of_date}.")
    inference_row = enriched[inference_mask].iloc[-1]
    candidate_train = enriched[enriched["date"] < pd.to_datetime(as_of_date).date()].copy()
    target_column = f"realized_risk_bucket_{int(target_horizon)}d"
    candidate_train = candidate_train.dropna(subset=[target_column]).reset_index(drop=True)

    feature_columns: list[str] = []
    for family_columns in feature_bundle.feature_families.values():
        for column in family_columns:
            if column in candidate_train.columns and column not in feature_columns:
                feature_columns.append(column)
    candidate_train = candidate_train.dropna(subset=feature_columns).reset_index(drop=True)

    fallback_required = len(candidate_train) < int(min_training_rows)
    return PathAwareDatasetBundle(
        training_frame=candidate_train,
        inference_row=inference_row,
        feature_columns=feature_columns,
        feature_families=feature_bundle.feature_families,
        warnings=list(feature_bundle.warnings),
        fallback_required=fallback_required,
    )


def build_path_aware_dataset_from_precomputed(
    *,
    cache: PathAwarePrecomputedCache,
    as_of_date: str,
    target_horizon: int,
    min_training_rows: int,
    train_lookback_days: int | None = None,
) -> PathAwareDatasetBundle:
    enriched = cache.enriched_frame
    as_of_text = str(pd.to_datetime(as_of_date).date())
    as_of_index = cache.date_to_last_index.get(as_of_text)
    if as_of_index is None:
        raise RuntimeError(f"Path-aware precompute cache has no row for as_of_date={as_of_date}.")

    inference_row = enriched.iloc[int(as_of_index)]
    target_column = f"realized_risk_bucket_{int(target_horizon)}d"

    feature_columns: list[str] = []
    for family_columns in cache.feature_families.values():
        for column in family_columns:
            if column in enriched.columns and column not in feature_columns:
                feature_columns.append(column)

    start_index = 0
    if train_lookback_days is not None and int(train_lookback_days) > 0:
        start_index = max(0, int(as_of_index) - int(train_lookback_days))
    candidate_train = enriched.iloc[start_index : int(as_of_index)].copy()
    candidate_train = candidate_train.dropna(subset=[target_column])
    candidate_train = candidate_train.dropna(subset=feature_columns).reset_index(drop=True)

    fallback_required = len(candidate_train) < int(min_training_rows)
    return PathAwareDatasetBundle(
        training_frame=candidate_train,
        inference_row=inference_row,
        feature_columns=feature_columns,
        feature_families=dict(cache.feature_families),
        warnings=list(cache.warnings),
        fallback_required=fallback_required,
    )
