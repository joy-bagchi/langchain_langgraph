from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.hmm_v4_path_aware_meta_agent import generate_hmm_v4_prediction
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


def _train_frame(rows: int = 320) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    dates = pd.bdate_range(end="2026-06-12", periods=rows).date
    states = ["STABLE_LOW_VOL_TREND", "MID_VOL_CHOP", "VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"]
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
            "regime_target": [states[int(i // 20) % len(states)] for i in idx],
        }
    )


def test_hmm_v4_path_aware_meta_runs_with_feature_importances(monkeypatch) -> None:
    import src.backtest.hmm_replay.replay_predictions as replay_predictions

    monkeypatch.setattr(replay_predictions, "GaussianHMM", FakeGaussianHMM)
    frame = _train_frame(320)
    context = create_replay_context(str(frame["date"].iloc[-1]))
    record = generate_prediction_record(
        context=context,
        model_name="hmm_v4_path_aware_meta",
        train_df=frame,
        min_train_rows=100,
        n_components=4,
        random_state=42,
        covariance_type="diag",
    )
    assert record["model_name"] == "hmm_v4_path_aware_meta"
    assert "path_aware_estimator" in record["model_diagnostics"]
    assert "path_features" in record["model_diagnostics"]
    if record["model_diagnostics"].get("fallback_used"):
        assert record["warnings"]
    else:
        assert isinstance(record["model_diagnostics"].get("top_feature_importances"), list)


def test_hmm_v4_path_aware_meta_falls_back_when_training_is_insufficient() -> None:
    frame = _train_frame(80)
    fallback_prediction = {
        "model_name": "hmm_v3_1_meta_blend",
        "top_state": "MID_VOL_CHOP",
        "state_probabilities": {"STABLE_LOW_VOL_TREND": 0.2, "MID_VOL_CHOP": 0.5, "VOL_EXPANSION_TRANSITION": 0.2, "HIGH_VOL_RISK_OFF": 0.1},
        "transition_probabilities": {"to_higher_vol_1d": 0.4, "to_higher_vol_2d": 0.4, "to_higher_vol_3d": 0.4},
        "policy_output": {"overwrite_posture": "LIGHT_OVERWRITE"},
        "model_diagnostics": {},
        "warnings": [],
    }
    result = generate_hmm_v4_prediction(
        as_of_date=str(frame["date"].iloc[-1]),
        train_df=frame,
        fallback_prediction=fallback_prediction,
    )
    assert result["model_name"] == "hmm_v4_path_aware_meta"
    assert result["model_diagnostics"]["fallback_used"] is True
    assert result["warnings"]
