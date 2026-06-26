from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.policy.policy_backtester import PolicyBacktestConfig, run_policy_backtest


def _sample_feature_store(path: Path) -> None:
    dates = pd.bdate_range("2025-01-01", periods=40)
    spot = 600.0
    rows = []
    for i, d in enumerate(dates):
        spot = spot + (1.5 if i % 5 else -2.0)
        rows.append(
            {
                "date": d.date(),
                "spy_close": spot,
                "vix": 16.0 + (i % 7),
                "vvix": 90.0 + (i % 11),
                "realized_vol_5d": 10.0 + (i % 6),
                "realized_vol_21d": 12.0 + (i % 8),
                "spy_return_1d": 0.002,
                "vvix_vix_ratio": 5.5,
                "vix9d_vix_ratio": 0.98,
                "vix_vix3m_ratio": 0.97,
                "term_structure_slope": 0.15,
                "drawdown_21d": -0.03,
                "trend_persistence_21d": 0.10,
                "avg_pairwise_corr_21d": 0.40,
                "first_eigenvalue_share_21d": 0.33,
                "effective_rank_21d": 5.0,
                "log_det_corr_21d": -0.5,
                "geometry_stress_score": 0.2,
                "regime_target": "MID_VOL_CHOP",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_policy_backtester_runs(tmp_path: Path) -> None:
    feature_store = tmp_path / "features.parquet"
    _sample_feature_store(feature_store)
    output_dir = tmp_path / "out"
    result = run_policy_backtest(
        config=PolicyBacktestConfig(
            feature_store_path=str(feature_store),
            output_dir=str(output_dir),
            run_mode="tuning",
            start_date="2025-01-01",
            end_date="2025-03-01",
            models=["heuristic"],
            train_lookback_days=30,
            min_train_rows=20,
        )
    )
    assert Path(result["policy_model_summary_path"]).exists()
    assert Path(result["policy_trades_path"]).exists()
    assert Path(result["policy_daily_pnl_path"]).exists()
    assert Path(result["policy_audit_starting_assumptions_path"]).exists()
    assert Path(result["policy_audit_first_20_daily_rows_path"]).exists()
    assert Path(result["policy_audit_first_20_trades_path"]).exists()
    assert Path(result["policy_invariant_checks_path"]).exists()
    assert Path(result["policy_profit_loss_explanation_path"]).exists()
    assert isinstance(result.get("model_economic_leaderboard"), list)

    daily = pd.read_csv(result["policy_daily_pnl_path"])
    required_daily_columns = {
        "date",
        "spy_close",
        "leap_open",
        "leap_premium_estimate",
        "leap_delta",
        "leap_exposure_pct",
        "leap_daily_pnl",
        "leap_cumulative_pnl",
        "short_call_open",
        "short_call_strike",
        "short_call_entry_premium",
        "short_call_mtm_value",
        "short_call_daily_pnl",
        "short_call_cumulative_pnl",
        "total_daily_pnl",
        "total_cumulative_pnl",
    }
    assert required_daily_columns.issubset(set(daily.columns))
    invariant_checks = pd.read_csv(result["policy_invariant_checks_path"])
    assert {"model_name", "check_name", "status", "details"}.issubset(set(invariant_checks.columns))
    assert "total_pnl_equals_leap_plus_overwrite" in set(invariant_checks["check_name"].astype(str))
    profit_loss = pd.read_csv(result["policy_profit_loss_explanation_path"])
    assert {"model_name", "leap_contribution", "overwrite_contribution"}.issubset(set(profit_loss.columns))


def test_policy_backtester_no_naked_short_calls_guardrail(tmp_path: Path) -> None:
    feature_store = tmp_path / "features.parquet"
    _sample_feature_store(feature_store)
    output_dir = tmp_path / "out_guardrail"
    result = run_policy_backtest(
        config=PolicyBacktestConfig(
            feature_store_path=str(feature_store),
            output_dir=str(output_dir),
            run_mode="tuning",
            start_date="2025-01-01",
            end_date="2025-03-01",
            models=["always_overwrite_baseline"],
            train_lookback_days=30,
            min_train_rows=20,
            leap_enabled=True,
            allow_leap_reentry=False,
            allow_naked_short_calls=False,
            leap_entry_premium=100.0,
            leap_profit_take_multiple=1.01,
            leap_stop_loss_multiple=0.99,
        )
    )
    trades = pd.read_csv(result["policy_trades_path"])
    leap_exits = trades[(trades["instrument_type"] == "LEAP") & (trades["exit_reason"].isin(["leap_profit_exit", "leap_stop_loss"]))]
    assert not leap_exits.empty
    if "entry_date" in leap_exits.columns:
        earliest_leap_exit = sorted(leap_exits["exit_date"].astype(str).tolist())[0]
        post_exit_short_calls = trades[
            (trades["instrument_type"] == "SHORT_CALL") & (trades["entry_date"].astype(str) > str(earliest_leap_exit))
        ]
        assert post_exit_short_calls.empty
