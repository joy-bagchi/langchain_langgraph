from __future__ import annotations

import pandas as pd

from src.backtest.policy.economic_metrics import build_policy_model_summary


def test_build_policy_model_summary_basic() -> None:
    trades_df = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "instrument_type": "SHORT_CALL",
                "dollar_pnl": 120.0,
                "entry_premium": 1.5,
                "exit_reason": "profit_exit",
                "holding_days": 1,
            },
            {
                "model_name": "m1",
                "instrument_type": "SHORT_CALL",
                "dollar_pnl": -150.0,
                "entry_premium": 1.5,
                "exit_reason": "loss_exit",
                "holding_days": 1,
            },
            {
                "model_name": "m1",
                "instrument_type": "LEAP",
                "dollar_pnl": 2000.0,
                "entry_premium": 100.0,
                "exit_reason": "leap_profit_exit",
                "holding_days": 20,
            },
        ]
    )
    daily_df = pd.DataFrame(
        [
            {"model_name": "m1", "total_daily_pnl": 3.0, "leap_daily_pnl": 2.0, "short_call_daily_pnl": 1.0, "leap_exposure_pct": 1.0},
            {"model_name": "m1", "total_daily_pnl": -1.0, "leap_daily_pnl": -0.5, "short_call_daily_pnl": -0.5, "leap_exposure_pct": 0.75},
        ]
    )
    missed_df = pd.DataFrame([{"model_name": "m1", "dollar_missed_risk": -3.0}])
    summary = build_policy_model_summary(trades_df=trades_df, daily_df=daily_df, missed_risk_df=missed_df)
    assert not summary.empty
    row = summary.iloc[0].to_dict()
    assert row["model_name"] == "m1"
    assert row["number_of_overwrites"] == 2
    assert row["total_pnl"] == 2.0
    assert row["leap_exit_profit_count"] == 1
