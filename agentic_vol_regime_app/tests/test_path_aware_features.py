from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.hmm_replay.path_aware_dataset import build_path_aware_dataset
from src.features.path_aware_features import build_path_aware_feature_frame


def _frame(rows: int = 320) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    dates = pd.bdate_range(end="2026-06-12", periods=rows).date
    states = ["STABLE_LOW_VOL_TREND", "MID_VOL_CHOP", "VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"]
    return pd.DataFrame(
        {
            "date": dates,
            "spy_close": 700.0 + idx * 0.3,
            "spy_return_1d": 0.0005 + (idx * 0.0),
            "realized_vol_5d": 12.0 + idx * 0.01,
            "realized_vol_21d": 14.0 + idx * 0.015,
            "vix": 15.0 + idx * 0.02,
            "vix_z_22d": idx * 0.001,
            "vvix": 90.0 + idx * 0.05,
            "vvix_vix_ratio": 5.5 + idx * 0.0004,
            "vvix_vix_z_22d": idx * 0.001,
            "vix9d_vix_ratio": 0.95 + idx * 0.0002,
            "vix_vix3m_ratio": 0.90 + idx * 0.0002,
            "term_structure_slope": 1.8 - idx * 0.001,
            "drawdown_21d": 0.03 + idx * 0.00005,
            "trend_persistence_21d": 0.55 + idx * 0.0001,
            "avg_pairwise_corr_21d": 0.20 + idx * 0.001,
            "first_eigenvalue_share_21d": 0.30 + idx * 0.001,
            "effective_rank_21d": 5.0 - idx * 0.005,
            "log_det_corr_21d": -1.5 - idx * 0.01,
            "regime_target": [states[min(3, int(i // 80))] for i in idx],
        }
    )


def test_path_aware_features_use_only_prior_rows_for_deltas() -> None:
    frame = _frame(120)
    bundle = build_path_aware_feature_frame(frame, feature_windows=[1, 3, 5, 10, 21, 63], geometry_stress_lookback=63)
    features = bundle.features
    row = features.iloc[50]
    expected = float(features.iloc[50]["geometry_stress_score"] - features.iloc[45]["geometry_stress_score"])
    assert abs(float(row["geometry_stress_score_delta_5d"]) - expected) < 1e-9


def test_path_aware_curvature_matches_definition() -> None:
    frame = _frame(120)
    bundle = build_path_aware_feature_frame(frame, feature_windows=[1, 3, 5, 10, 21, 63], geometry_stress_lookback=63)
    features = bundle.features.reset_index(drop=True)
    idx = 40
    expected = (
        (features.loc[idx, "geometry_stress_score"] - features.loc[idx - 5, "geometry_stress_score"]) / 5.0
        - (features.loc[idx - 5, "geometry_stress_score"] - features.loc[idx - 10, "geometry_stress_score"]) / 5.0
    )
    assert abs(float(features.loc[idx, "geometry_stress_score_curvature_5_10"]) - float(expected)) < 1e-9


def test_path_aware_persistence_counts_are_computed() -> None:
    frame = _frame(120)
    bundle = build_path_aware_feature_frame(frame, feature_windows=[1, 3, 5, 10, 21, 63], geometry_stress_lookback=63)
    features = bundle.features
    assert "geometry_days_above_0_55_21d" in features.columns
    assert float(features.iloc[-1]["geometry_days_above_0_55_21d"]) >= 0.0


def test_vol_geometry_gap_is_computed() -> None:
    frame = _frame(120)
    bundle = build_path_aware_feature_frame(frame, feature_windows=[1, 3, 5, 10, 21, 63], geometry_stress_lookback=63)
    features = bundle.features
    row = features.iloc[-1]
    assert abs(float(row["vol_geometry_gap"]) - (float(row["core_vol_risk_score"]) - float(row["geometry_stress_score"]))) < 1e-9


def test_walk_forward_dataset_excludes_as_of_target_rows() -> None:
    frame = _frame(320)
    as_of = str(frame.iloc[-1]["date"])
    dataset = build_path_aware_dataset(
        frame,
        as_of_date=as_of,
        target_horizon=3,
        feature_windows=[1, 3, 5, 10, 21, 63],
        geometry_stress_lookback=63,
        min_training_rows=100,
    )
    assert dataset.training_frame["date"].max() < pd.to_datetime(as_of).date()
    assert dataset.fallback_required is False
