from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_sharpe_like(daily_pnl: pd.Series) -> float:
    mean = float(daily_pnl.mean()) if not daily_pnl.empty else 0.0
    std = float(daily_pnl.std(ddof=0)) if not daily_pnl.empty else 0.0
    if std <= 1e-9:
        return 0.0
    return mean / std


def _max_drawdown(daily_pnl: pd.Series) -> float:
    if daily_pnl.empty:
        return 0.0
    cumulative = daily_pnl.cumsum()
    rolling_peak = cumulative.cummax()
    drawdown = cumulative - rolling_peak
    return float(drawdown.min())


def build_policy_model_summary(
    *,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    missed_risk_df: pd.DataFrame,
) -> pd.DataFrame:
    if "model_name" not in trades_df.columns:
        trades_df = pd.DataFrame(columns=["model_name"])
    if "model_name" not in daily_df.columns:
        daily_df = pd.DataFrame(columns=["model_name"])
    if "model_name" not in missed_risk_df.columns:
        missed_risk_df = pd.DataFrame(columns=["model_name", "dollar_missed_risk"])
    model_names = sorted(set(daily_df.get("model_name", pd.Series(dtype=str)).unique().tolist()))
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        scoped_daily = daily_df[daily_df["model_name"] == model_name].copy()
        scoped_trades = trades_df[trades_df["model_name"] == model_name].copy()
        scoped_missed = missed_risk_df[missed_risk_df["model_name"] == model_name].copy()

        daily_pnl = pd.to_numeric(scoped_daily.get("total_daily_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        leap_daily = pd.to_numeric(scoped_daily.get("leap_daily_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        overwrite_daily = pd.to_numeric(
            scoped_daily.get("short_call_daily_pnl", scoped_daily.get("overwrite_daily_pnl", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0.0)

        if "instrument_type" not in scoped_trades.columns:
            scoped_trades["instrument_type"] = "SHORT_CALL"
        short_call_trades = scoped_trades[scoped_trades["instrument_type"] == "SHORT_CALL"].copy()
        leap_trades = scoped_trades[scoped_trades["instrument_type"] == "LEAP"].copy()
        trade_option_pnl = pd.to_numeric(
            short_call_trades.get("dollar_pnl", short_call_trades.get("option_pnl", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0.0)
        trade_total_pnl = pd.to_numeric(
            scoped_trades.get("dollar_pnl", scoped_trades.get("total_trade_pnl", pd.Series(dtype=float))),
            errors="coerce",
        ).fillna(0.0)
        entry_premium = pd.to_numeric(short_call_trades.get("entry_premium", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        days_held = pd.to_numeric(scoped_trades.get("holding_days", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        premium_collected = float((entry_premium * 100.0).sum())
        premium_retained = float(trade_option_pnl[trade_option_pnl > 0].sum())
        premium_retained_pct = (premium_retained / premium_collected) if premium_collected > 1e-9 else 0.0

        loss_exits = short_call_trades[short_call_trades["exit_reason"] == "loss_exit"]
        profit_exits = short_call_trades[
            short_call_trades["exit_reason"].isin(["profit_exit", "roll_down_profit_capture"])
        ]

        total_pnl = float(daily_pnl.sum())
        leap_pnl = float(leap_daily.sum())
        overwrite_pnl = float(overwrite_daily.sum())
        rows.append(
            {
                "model_name": model_name,
                "total_pnl": total_pnl,
                "leap_pnl": leap_pnl,
                "overwrite_pnl": overwrite_pnl,
                "total_return_pct": 0.0,
                "max_drawdown": _max_drawdown(daily_pnl),
                "worst_day_pnl": float(daily_pnl.min()) if not daily_pnl.empty else 0.0,
                "worst_trade_pnl": float(trade_total_pnl.min()) if not trade_total_pnl.empty else 0.0,
                "average_daily_pnl": float(daily_pnl.mean()) if not daily_pnl.empty else 0.0,
                "median_daily_pnl": float(daily_pnl.median()) if not daily_pnl.empty else 0.0,
                "volatility_of_daily_pnl": float(daily_pnl.std(ddof=0)) if not daily_pnl.empty else 0.0,
                "sharpe_like_score": _safe_sharpe_like(daily_pnl),
                "number_of_overwrites": int(len(short_call_trades)),
                "overwrite_win_rate": float((trade_option_pnl > 0).mean()) if len(trade_option_pnl) else 0.0,
                "average_overwrite_pnl": float(trade_option_pnl.mean()) if len(trade_option_pnl) else 0.0,
                "median_overwrite_pnl": float(trade_option_pnl.median()) if len(trade_option_pnl) else 0.0,
                "premium_collected": premium_collected,
                "premium_retained_pct": premium_retained_pct,
                "average_premium_collected": float((entry_premium * 100.0).mean()) if len(entry_premium) else 0.0,
                "average_premium_lost_on_loss_exit": float(
                    pd.to_numeric(loss_exits.get("dollar_pnl", loss_exits.get("option_pnl", pd.Series(dtype=float))), errors="coerce").mean()
                )
                if not loss_exits.empty
                else 0.0,
                "number_profit_exits": int(len(profit_exits)),
                "number_loss_exits": int(len(loss_exits)),
                "number_underlying_touch_exits": int((short_call_trades["exit_reason"] == "underlying_touch").sum())
                if not short_call_trades.empty
                else 0,
                "number_expirations": int((short_call_trades["exit_reason"] == "expiration").sum())
                if not short_call_trades.empty
                else 0,
                "average_holding_days": float(days_held.mean()) if len(days_held) else 0.0,
                "tail_loss_1pct": float(daily_pnl.quantile(0.01)) if len(daily_pnl) else 0.0,
                "tail_loss_5pct": float(daily_pnl.quantile(0.05)) if len(daily_pnl) else 0.0,
                "catastrophic_trade_count": int((trade_total_pnl <= -200.0).sum()) if len(trade_total_pnl) else 0,
                "average_loss_when_model_missed_risk": float(
                    pd.to_numeric(scoped_missed.get("dollar_missed_risk", pd.Series(dtype=float)), errors="coerce").mean()
                )
                if not scoped_missed.empty
                else 0.0,
                "dollar_missed_risk_score": float(
                    pd.to_numeric(scoped_missed.get("dollar_missed_risk", pd.Series(dtype=float)), errors="coerce").sum()
                )
                if not scoped_missed.empty
                else 0.0,
                "leap_exposure_avg": float(
                    pd.to_numeric(scoped_daily.get("leap_exposure_pct", pd.Series(dtype=float)), errors="coerce").mean()
                )
                if not scoped_daily.empty
                else 0.0,
                "leap_exposure_days_full": int(
                    (pd.to_numeric(scoped_daily.get("leap_exposure_pct", pd.Series(dtype=float)), errors="coerce") >= 0.99).sum()
                )
                if not scoped_daily.empty
                else 0,
                "leap_exposure_days_reduced": int(
                    (pd.to_numeric(scoped_daily.get("leap_exposure_pct", pd.Series(dtype=float)), errors="coerce") < 0.99).sum()
                )
                if not scoped_daily.empty
                else 0,
                "leap_exit_profit_count": int((leap_trades["exit_reason"] == "leap_profit_exit").sum()) if not leap_trades.empty else 0,
                "leap_exit_stop_count": int((leap_trades["exit_reason"] == "leap_stop_loss").sum()) if not leap_trades.empty else 0,
                "avoided_leap_drawdown_estimate": float(
                    pd.to_numeric(scoped_daily.get("avoided_leap_drawdown_estimate", pd.Series(dtype=float)), errors="coerce").sum()
                )
                if not scoped_daily.empty
                else 0.0,
                "missed_leap_upside_estimate": float(
                    pd.to_numeric(scoped_daily.get("missed_leap_upside_estimate", pd.Series(dtype=float)), errors="coerce").sum()
                )
                if not scoped_daily.empty
                else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
