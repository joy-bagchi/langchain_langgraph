from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.hmm_replay.replay_predictions import create_replay_context, generate_prediction_record


def test_replay_rejects_training_rows_after_as_of_date() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2026-06-10").date(),
                "spy_close": 750.0,
                "spy_return_1d": 0.001,
                "vix": 15.0,
                "vvix_vix_ratio": 5.4,
                "realized_vol_5d": 12.0,
                "realized_vol_21d": 14.0,
                "term_structure_slope": 1.2,
                "drawdown_21d": 0.03,
                "trend_persistence_21d": 0.6,
            },
            {
                "date": pd.to_datetime("2026-06-11").date(),
                "spy_close": 752.0,
                "spy_return_1d": 0.002,
                "vix": 14.8,
                "vvix_vix_ratio": 5.3,
                "realized_vol_5d": 11.8,
                "realized_vol_21d": 13.8,
                "term_structure_slope": 1.1,
                "drawdown_21d": 0.02,
                "trend_persistence_21d": 0.62,
            },
        ]
    )
    context = create_replay_context("2026-06-10")

    with pytest.raises(RuntimeError, match="no-lookahead violation"):
        generate_prediction_record(
            context=context,
            model_name="heuristic",
            train_df=frame,
            min_train_rows=1,
            n_components=4,
            random_state=42,
            covariance_type="diag",
        )
