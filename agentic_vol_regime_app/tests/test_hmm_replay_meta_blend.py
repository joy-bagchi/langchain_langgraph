from __future__ import annotations

import numpy as np
import pandas as pd

import src.backtest.hmm_replay.replay_predictions as replay_predictions
from src.backtest.hmm_replay.replay_predictions import create_replay_context, generate_prediction_record


class FakeGaussianHMM:
    def __init__(self, *, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
        self.n_components = n_components
        self.transmat_ = np.asarray(
            [
                [0.8, 0.1, 0.07, 0.03],
                [0.1, 0.7, 0.15, 0.05],
                [0.06, 0.16, 0.62, 0.16],
                [0.03, 0.07, 0.18, 0.72],
            ],
            dtype=float,
        )
        self.means_ = None

    def fit(self, values: np.ndarray) -> "FakeGaussianHMM":
        self.means_ = np.tile(np.linspace(-0.8, 0.8, self.n_components).reshape(-1, 1), (1, values.shape[1]))
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.asarray([[0.18, 0.24, 0.34, 0.24] for _ in range(values.shape[0])], dtype=float)


def _train_frame(rows: int = 200) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    dates = pd.bdate_range(end="2026-06-12", periods=rows).date
    return pd.DataFrame(
        {
            "date": dates,
            "spy_close": 700.0 + idx * 0.5,
            "spy_return_1d": 0.0005 + idx * 0.0,
            "realized_vol_5d": 12.0 + idx * 0.01,
            "realized_vol_21d": 14.0 + idx * 0.01,
            "vix": 15.0 + idx * 0.01,
            "vix_z_22d": idx * 0.001,
            "vvix": 90.0 + idx * 0.01,
            "vvix_vix_ratio": 5.5 + idx * 0.0001,
            "vvix_vix_z_22d": idx * 0.001,
            "vix9d_vix_ratio": 0.95 + idx * 0.0001,
            "vix_vix3m_ratio": 0.9 + idx * 0.0001,
            "term_structure_slope": 1.2 - idx * 0.0005,
            "drawdown_21d": 0.03 + idx * 0.0,
            "trend_persistence_21d": 0.6 + idx * 0.0,
            "avg_pairwise_corr_21d": 0.25 + idx * 0.001,
            "first_eigenvalue_share_21d": 0.35 + idx * 0.001,
            "effective_rank_21d": 5.0 - idx * 0.01,
            "log_det_corr_21d": -1.0 - idx * 0.01,
        }
    )


def test_replay_supports_hmm_v3_1_meta_blend(monkeypatch) -> None:
    monkeypatch.setattr(replay_predictions, "GaussianHMM", FakeGaussianHMM)
    frame = _train_frame(220)
    context = create_replay_context(str(frame["date"].iloc[-1]))

    meta_record = generate_prediction_record(
        context=context,
        model_name="hmm_v3_1_meta_blend",
        train_df=frame,
        min_train_rows=100,
        n_components=4,
        random_state=42,
        covariance_type="diag",
    )
    v1_record = generate_prediction_record(
        context=context,
        model_name="hmm_v1_core",
        train_df=frame,
        min_train_rows=100,
        n_components=4,
        random_state=42,
        covariance_type="diag",
    )
    v3_record = generate_prediction_record(
        context=context,
        model_name="hmm_v3_core_plus_sector_geometry",
        train_df=frame,
        min_train_rows=100,
        n_components=4,
        random_state=42,
        covariance_type="diag",
    )

    assert meta_record["model_name"] == "hmm_v3_1_meta_blend"
    assert "geometry_stress_score" in meta_record["model_diagnostics"]
    assert "core_vol_risk_score" in meta_record["model_diagnostics"]
    assert meta_record["top_state"] in {
        "STABLE_LOW_VOL_TREND",
        "MID_VOL_CHOP",
        "VOL_EXPANSION_TRANSITION",
        "HIGH_VOL_RISK_OFF",
    }
    assert v1_record["model_name"] == "hmm_v1_core"
    assert v3_record["model_name"] == "hmm_v3_core_plus_sector_geometry"
