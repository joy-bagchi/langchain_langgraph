from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtest.hmm_replay.replay_dataset import filter_date_range, load_feature_store
from src.backtest.hmm_replay.replay_predictions import create_replay_context, generate_prediction_record
from src.backtest.policy.economic_metrics import build_policy_model_summary
from src.backtest.policy.leap_policy import leap_exposure_from_brace_score, overwrite_action_from_brace_score
from src.backtest.policy.option_pricing_proxy import black_scholes_call_price
from src.backtest.policy.overwrite_policy import (
    candidate_short_call_strike,
    nearest_listed_strike,
    vix_premium_target,
)

TUNING_WINDOW_BDAYS = 90
TUNING_MAX_DATES = 75
MODEL_OPTION_IV_MULTIPLIER = 1.0


@dataclass(slots=True)
class PolicyBacktestConfig:
    feature_store_path: str = "agentic_vol_regime_app/data/processed/features_daily.parquet"
    output_dir: str = "reports/backtests/policy_backtest/"
    run_mode: str = "tuning"
    start_date: str = "2024-01-01"
    end_date: str = "latest"
    models: list[str] | None = None
    train_lookback_days: int = 756
    min_train_rows: int = 504
    default_dte: int = 1
    strike_increment: float = 1.0
    leap_enabled: bool = True
    leap_contracts: int = 1
    leap_delta: float = 0.75
    leap_multiplier: int = 100
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    profit_exit_pct: float = 0.20
    loss_exit_multiple: float = 2.0
    exit_on_underlying_touch: bool = True
    safer_reference_mode: str = "no_overwrite"
    leap_entry_premium: float = 100.0
    leap_profit_take_multiple: float = 1.20
    leap_stop_loss_multiple: float = 0.80
    allow_leap_reentry: bool = True
    allow_naked_short_calls: bool = False


def _round_columns(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(digits)
    return frame


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _build_model_starting_assumptions(
    *,
    model_names: list[str],
    scoped: pd.DataFrame,
    config: PolicyBacktestConfig,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    initial_spy = float(scoped.iloc[0]["spy_close"]) if not scoped.empty and "spy_close" in scoped.columns else 0.0
    rows = [
        {
            "model_name": model_name,
            "starting_cash": 0.0,
            "leap_enabled": bool(config.leap_enabled),
            "leap_entry_premium": float(config.leap_entry_premium),
            "leap_delta": float(config.leap_delta),
            "leap_multiplier": int(config.leap_multiplier),
            "short_call_multiplier": 100,
            "allow_naked_short_calls": bool(config.allow_naked_short_calls),
            "initial_spy": round(float(initial_spy), 2),
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
        for model_name in model_names
    ]
    return pd.DataFrame(rows)


def _build_policy_mechanics_audit_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    working = daily_df.copy()
    working["spy_change"] = pd.to_numeric(working.get("spy_close"), errors="coerce").diff().fillna(0.0)
    working["action_taken"] = "hold"
    if "short_call_open" in working.columns:
        working.loc[working["short_call_open"] == True, "action_taken"] = "short_call_active"  # noqa: E712
    if "leap_open" in working.columns:
        working.loc[(working["leap_open"] == False) & (working["short_call_open"] == False), "action_taken"] = "cash_or_flat"  # noqa: E712
    working["exit_reason"] = ""
    if "model_name" not in working.columns:
        return pd.DataFrame()
    desired_columns = [
        "model_name",
        "date",
        "spy_close",
        "spy_change",
        "leap_open",
        "leap_premium_estimate",
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
        "action_taken",
        "exit_reason",
    ]
    available = [column for column in desired_columns if column in working.columns]
    return (
        working.groupby("model_name", group_keys=False)
        .head(20)[available]
        .reset_index(drop=True)
    )


def _build_policy_mechanics_audit_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    desired_columns = [
        "model_name",
        "instrument_type",
        "entry_date",
        "exit_date",
        "entry_spy",
        "exit_spy",
        "entry_premium",
        "exit_premium",
        "multiplier",
        "dollar_pnl",
        "exit_reason",
    ]
    available = [column for column in desired_columns if column in trades_df.columns]
    return trades_df.groupby("model_name", group_keys=False).head(20)[available].reset_index(drop=True)


def _build_policy_invariant_checks(
    *,
    model_summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    config: PolicyBacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_names = sorted(set(model_summary_df.get("model_name", pd.Series(dtype=str)).astype(str).tolist()))
    for model_name in model_names:
        summary = model_summary_df[model_summary_df["model_name"] == model_name]
        scoped_trades = trades_df[trades_df.get("model_name", pd.Series(dtype=str)) == model_name].copy()
        scoped_daily = daily_df[daily_df.get("model_name", pd.Series(dtype=str)) == model_name].copy()
        total_pnl = _safe_float(summary.iloc[0]["total_pnl"]) if not summary.empty else 0.0
        leap_pnl = _safe_float(summary.iloc[0]["leap_pnl"]) if not summary.empty else 0.0
        overwrite_pnl = _safe_float(summary.iloc[0]["overwrite_pnl"]) if not summary.empty else 0.0
        total_return_pct = _safe_float(summary.iloc[0]["total_return_pct"]) if not summary.empty else 0.0
        short_call_trades = scoped_trades[scoped_trades.get("instrument_type", pd.Series(dtype=str)) == "SHORT_CALL"].copy()
        all_option_multiplier_100 = bool(short_call_trades.empty or (pd.to_numeric(short_call_trades.get("multiplier"), errors="coerce") == 100).all())
        naked_short_count = 0
        if {"leap_open", "short_call_open"}.issubset(scoped_daily.columns):
            naked_short_count = int(((scoped_daily["leap_open"] == False) & (scoped_daily["short_call_open"] == True)).sum())  # noqa: E712
        rows.extend(
            [
                {
                    "model_name": model_name,
                    "check_name": "total_pnl_equals_leap_plus_overwrite",
                    "status": bool(abs(float(total_pnl or 0.0) - (float(leap_pnl or 0.0) + float(overwrite_pnl or 0.0))) < 0.05),
                    "details": f"total={total_pnl}, leap={leap_pnl}, overwrite={overwrite_pnl}",
                },
                {
                    "model_name": model_name,
                    "check_name": "no_overwrite_baseline_has_zero_overwrite_pnl",
                    "status": bool(model_name != "no_overwrite_baseline" or abs(float(overwrite_pnl or 0.0)) < 0.05),
                    "details": f"overwrite_pnl={overwrite_pnl}",
                },
                {
                    "model_name": model_name,
                    "check_name": "no_overwrite_baseline_has_no_short_call_trades",
                    "status": bool(model_name != "no_overwrite_baseline" or short_call_trades.empty),
                    "details": f"short_call_trade_count={len(short_call_trades)}",
                },
                {
                    "model_name": model_name,
                    "check_name": "nonzero_total_pnl_requires_nonzero_total_return_or_missing_denominator",
                    "status": bool(abs(float(total_pnl or 0.0)) < 0.05 or abs(float(total_return_pct or 0.0)) > 1e-9),
                    "details": f"total_pnl={total_pnl}, total_return_pct={total_return_pct}",
                },
                {
                    "model_name": model_name,
                    "check_name": "all_option_pnl_uses_multiplier_100",
                    "status": all_option_multiplier_100,
                    "details": f"short_call_trade_count={len(short_call_trades)}",
                },
                {
                    "model_name": model_name,
                    "check_name": "short_calls_not_open_when_leap_closed_unless_allowed",
                    "status": bool(config.allow_naked_short_calls or naked_short_count == 0),
                    "details": f"naked_short_daily_rows={naked_short_count}, allow_naked_short_calls={config.allow_naked_short_calls}",
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_policy_profit_loss_explanation(
    *,
    model_summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in sorted(set(model_summary_df.get("model_name", pd.Series(dtype=str)).astype(str).tolist())):
        summary = model_summary_df[model_summary_df["model_name"] == model_name]
        scoped_trades = trades_df[trades_df.get("model_name", pd.Series(dtype=str)) == model_name].copy()
        scoped_daily = daily_df[daily_df.get("model_name", pd.Series(dtype=str)) == model_name].copy()
        short_call_trades = scoped_trades[scoped_trades.get("instrument_type", pd.Series(dtype=str)) == "SHORT_CALL"].copy()
        leap_drawdown = 0.0
        if "leap_cumulative_pnl" in scoped_daily.columns and not scoped_daily.empty:
            leap_curve = pd.to_numeric(scoped_daily["leap_cumulative_pnl"], errors="coerce").fillna(0.0)
            leap_drawdown = float((leap_curve - leap_curve.cummax()).min())
        rows.append(
            {
                "model_name": model_name,
                "leap_contribution": _safe_float(summary.iloc[0]["leap_pnl"]) if not summary.empty else 0.0,
                "overwrite_contribution": _safe_float(summary.iloc[0]["overwrite_pnl"]) if not summary.empty else 0.0,
                "number_of_short_calls_sold": int(len(short_call_trades)),
                "number_of_profit_exits": int((short_call_trades.get("exit_reason", pd.Series(dtype=str)) == "profit_exit").sum()) if not short_call_trades.empty else 0,
                "number_of_loss_exits": int((short_call_trades.get("exit_reason", pd.Series(dtype=str)) == "loss_exit").sum()) if not short_call_trades.empty else 0,
                "number_of_touch_exits": int((short_call_trades.get("exit_reason", pd.Series(dtype=str)) == "underlying_touch").sum()) if not short_call_trades.empty else 0,
                "average_short_call_pnl": round(float(pd.to_numeric(short_call_trades.get("dollar_pnl", pd.Series(dtype=float)), errors="coerce").mean()), 2) if not short_call_trades.empty else 0.0,
                "worst_short_call_loss": round(float(pd.to_numeric(short_call_trades.get("dollar_pnl", pd.Series(dtype=float)), errors="coerce").min()), 2) if not short_call_trades.empty else 0.0,
                "worst_leap_drawdown": round(float(leap_drawdown), 2),
            }
        )
    return pd.DataFrame(rows)


def load_policy_backtest_config(path: str | Path) -> PolicyBacktestConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PolicyBacktestConfig(
        feature_store_path=str(
            payload.get("feature_store_path", "agentic_vol_regime_app/data/processed/features_daily.parquet")
        ),
        output_dir=str(payload.get("output_dir", "reports/backtests/policy_backtest/")),
        run_mode=str(payload.get("run_mode", "tuning")).strip().lower() or "tuning",
        start_date=str(payload.get("start_date", "2024-01-01")),
        end_date=str(payload.get("end_date", "latest")),
        models=[str(item) for item in list(payload.get("models", ["hmm_v3_1_meta_blend", "hmm_v4_path_aware_meta"]))],
        train_lookback_days=int(payload.get("train_lookback_days", 756)),
        min_train_rows=int(payload.get("min_train_rows", 504)),
        default_dte=int(payload.get("default_dte", 1)),
        strike_increment=float(payload.get("strike_increment", 1.0)),
        leap_enabled=bool(payload.get("leap_enabled", True)),
        leap_contracts=int(payload.get("leap_contracts", 1)),
        leap_delta=float(payload.get("leap_delta", 0.75)),
        leap_multiplier=int(payload.get("leap_multiplier", 100)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.0)),
        dividend_yield=float(payload.get("dividend_yield", 0.0)),
        profit_exit_pct=float(payload.get("profit_exit_pct", 0.20)),
        loss_exit_multiple=float(payload.get("loss_exit_multiple", 2.0)),
        exit_on_underlying_touch=bool(payload.get("exit_on_underlying_touch", True)),
        safer_reference_mode=str(payload.get("safer_reference_mode", "no_overwrite")),
        leap_entry_premium=float(payload.get("leap_entry_premium", 100.0)),
        leap_profit_take_multiple=float(payload.get("leap_profit_take_multiple", 1.20)),
        leap_stop_loss_multiple=float(payload.get("leap_stop_loss_multiple", 0.80)),
        allow_leap_reentry=bool(payload.get("allow_leap_reentry", True)),
        allow_naked_short_calls=bool(payload.get("allow_naked_short_calls", False)),
    )


def _resolve_models(requested: list[str] | None) -> list[str]:
    base = ["no_overwrite_baseline", "always_overwrite_baseline", "heuristic_policy"]
    model_candidates = requested or ["hmm_v3_1_meta_blend", "hmm_v4_path_aware_meta"]
    ordered: list[str] = []
    for name in [*base, *model_candidates]:
        token = str(name).strip()
        if token and token not in ordered:
            ordered.append(token)
    return ordered


def _effective_dates(source: pd.DataFrame, start_date: str, end_date: str, run_mode: str) -> tuple[str, str]:
    end = str(pd.Timestamp.utcnow().date()) if str(end_date).strip().lower() == "latest" else str(end_date)
    start = str(start_date)
    if str(run_mode).strip().lower() == "tuning":
        latest = pd.to_datetime(source["date"].max()).date()
        tuning_start = pd.bdate_range(end=latest, periods=TUNING_WINDOW_BDAYS).min().date()
        start = str(max(pd.to_datetime(start).date(), tuning_start))
        end = str(latest)
    return start, end


def _brace_from_prediction(prediction: dict[str, Any]) -> float:
    probs = dict(prediction.get("state_probabilities", {}))
    return float(probs.get("VOL_EXPANSION_TRANSITION", 0.0)) + float(probs.get("HIGH_VOL_RISK_OFF", 0.0))


def _recommendation_for_row(
    *,
    model_name: str,
    row: pd.Series,
    prediction: dict[str, Any] | None,
    config: PolicyBacktestConfig,
) -> dict[str, Any]:
    vix = float(row.get("vix", 18.0))
    spot = float(row.get("spy_close", 0.0))
    premium = vix_premium_target(vix)
    expected_move = candidate_short_call_strike(spot, vix)
    selected_strike = nearest_listed_strike(expected_move, increment=config.strike_increment)
    dte = int(config.default_dte)

    if model_name == "no_overwrite_baseline":
        action = "none"
        brace = 0.0
        leap_exposure = 1.0
    elif model_name == "always_overwrite_baseline":
        action = "normal"
        brace = 0.45
        leap_exposure = 1.0
    elif model_name == "heuristic_policy":
        action = "none" if vix < 17.0 and float(row.get("term_structure_slope", 0.0)) > 0.0 else "normal"
        brace = 0.20 if action == "none" else 0.45
        leap_exposure = 1.0 if brace < 0.30 else 0.75
    else:
        resolved = prediction or {}
        brace = _brace_from_prediction(resolved)
        action = overwrite_action_from_brace_score(brace)
        leap_exposure = leap_exposure_from_brace_score(brace)

    if action == "conservative":
        premium_mid = premium.low
    elif action == "aggressive":
        premium_mid = premium.high
    else:
        premium_mid = premium.mid

    if action == "none":
        dte = 0

    return {
        "model_name": model_name,
        "as_of_date": str(row.get("date")),
        "predicted_regime": str((prediction or {}).get("top_state", row.get("regime_target", "STABLE_LOW_VOL_TREND"))),
        "brace_for_impact_score": round(float(brace), 6),
        "leap_exposure_pct": round(float(leap_exposure), 6),
        "overwrite_action": str(action),
        "target_premium_low": float(premium.low),
        "target_premium_high": float(premium.high),
        "target_premium_mid": float(premium_mid),
        "recommended_dte": int(dte),
        "expected_upside_move_pct": float((selected_strike / max(spot, 1e-6)) - 1.0),
        "candidate_strike": float(expected_move),
        "selected_strike": float(selected_strike),
        "profit_exit_pct": float(config.profit_exit_pct),
        "loss_exit_multiple": float(config.loss_exit_multiple),
        "exit_on_underlying_touch": bool(config.exit_on_underlying_touch),
    }


def _simulate_one_model(
    *,
    model_name: str,
    scoped: pd.DataFrame,
    full_source: pd.DataFrame,
    config: PolicyBacktestConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    missed_risk: list[dict[str, Any]] = []
    short_call: dict[str, Any] | None = None
    previous_close: float | None = None

    leap_open = bool(config.leap_enabled)
    leap_entry_date = str(scoped.iloc[0]["date"]) if leap_open and not scoped.empty else ""
    leap_entry_premium = float(config.leap_entry_premium)
    leap_premium_estimate = float(leap_entry_premium) if leap_open else 0.0
    leap_cumulative_pnl = 0.0
    leap_trade_cumulative_pnl = 0.0
    short_call_cumulative_pnl = 0.0
    total_cumulative_pnl = 0.0
    current_exposure = 1.0 if leap_open else 0.0
    pending_exposure = current_exposure
    reentry_next_day = False

    for row in scoped.itertuples(index=False):
        as_of = pd.to_datetime(getattr(row, "date")).date()
        as_of_text = as_of.isoformat()
        spot = float(getattr(row, "spy_close"))
        vix = float(getattr(row, "vix"))
        vvix = float(getattr(row, "vvix"))
        geometry_stress = float(getattr(row, "geometry_stress_score", 0.0)) if hasattr(row, "geometry_stress_score") else 0.0

        if reentry_next_day and config.allow_leap_reentry:
            leap_open = True
            leap_entry_date = as_of_text
            leap_entry_premium = float(config.leap_entry_premium)
            leap_premium_estimate = float(leap_entry_premium)
            leap_trade_cumulative_pnl = 0.0
            current_exposure = max(current_exposure, 1.0)
            pending_exposure = current_exposure
            reentry_next_day = False

        # 1) LEAP daily mark
        leap_daily_pnl = 0.0
        if leap_open and previous_close is not None:
            spot_change = spot - previous_close
            leap_daily_pnl = (
                float(config.leap_delta)
                * float(config.leap_multiplier)
                * float(config.leap_contracts)
                * float(current_exposure)
                * spot_change
            )
        leap_cumulative_pnl += float(leap_daily_pnl)
        leap_trade_cumulative_pnl += float(leap_daily_pnl)
        if leap_open:
            leap_premium_estimate = float(leap_entry_premium) + (
                leap_trade_cumulative_pnl / (float(config.leap_multiplier) * float(config.leap_contracts))
            )
        else:
            leap_premium_estimate = 0.0

        leap_exit_reason: str | None = None
        if leap_open:
            if leap_premium_estimate >= (float(leap_entry_premium) * float(config.leap_profit_take_multiple)):
                leap_exit_reason = "leap_profit_exit"
            elif leap_premium_estimate <= (float(leap_entry_premium) * float(config.leap_stop_loss_multiple)):
                leap_exit_reason = "leap_stop_loss"
        if leap_exit_reason:
            leap_trade_dollar_pnl = (float(leap_premium_estimate) - float(leap_entry_premium)) * float(config.leap_multiplier)
            trades.append(
                {
                    "model_name": model_name,
                    "instrument_type": "LEAP",
                    "entry_date": leap_entry_date,
                    "exit_date": as_of_text,
                    "entry_premium": float(leap_entry_premium),
                    "exit_premium": float(leap_premium_estimate),
                    "multiplier": int(config.leap_multiplier),
                    "dollar_pnl": float(leap_trade_dollar_pnl),
                    "exit_reason": leap_exit_reason,
                }
            )
            leap_open = False
            current_exposure = 0.0
            pending_exposure = 0.0
            leap_trade_cumulative_pnl = 0.0
            reentry_next_day = bool(config.allow_leap_reentry)

        # 2) Short call MTM/exit
        short_call_daily_pnl = 0.0
        short_call_open = bool(short_call is not None)
        short_call_strike = float(short_call["strike"]) if short_call is not None else None
        short_call_entry_premium = float(short_call["entry_premium"]) if short_call is not None else None
        short_call_mtm_value = float(short_call["last_option_value"]) if short_call is not None else None

        if short_call is not None:
            days_left = max(int(short_call["days_left"]), 0)
            iv = max(vix / 100.0 * MODEL_OPTION_IV_MULTIPLIER, 0.01)
            option_value = black_scholes_call_price(
                spot=spot,
                strike=float(short_call["strike"]),
                dte=days_left,
                iv_annual=iv,
                risk_free_rate=config.risk_free_rate,
                dividend_yield=config.dividend_yield,
            )
            prev_value = float(short_call["last_option_value"])
            short_call_daily_pnl = (prev_value - option_value) * float(config.leap_multiplier)
            short_call["last_option_value"] = float(option_value)
            short_call["days_left"] = max(days_left - 1, 0)
            short_call_mtm_value = float(option_value)

            exit_reason: str | None = None
            entry_premium = float(short_call["entry_premium"])
            if bool(short_call["exit_on_underlying_touch"]) and spot >= float(short_call["strike"]):
                exit_reason = "underlying_touch"
            elif option_value <= (entry_premium * float(short_call["profit_exit_pct"])):
                exit_reason = "profit_exit"
            elif option_value >= (entry_premium * float(short_call["loss_exit_multiple"])):
                exit_reason = "loss_exit"
            elif int(short_call["days_left"]) <= 0:
                exit_reason = "expiration"

            if exit_reason:
                exit_value = float(option_value) if exit_reason != "expiration" else max(spot - float(short_call["strike"]), 0.0)
                option_pnl_dollars = (entry_premium - exit_value) * float(config.leap_multiplier)
                trades.append(
                    {
                        "model_name": model_name,
                        "instrument_type": "SHORT_CALL",
                        "entry_date": str(short_call["entry_date"]),
                        "exit_date": as_of_text,
                        "entry_premium": float(entry_premium),
                        "exit_premium": float(exit_value),
                        "multiplier": int(config.leap_multiplier),
                        "dollar_pnl": float(option_pnl_dollars),
                        "exit_reason": str(exit_reason),
                        "strike": float(short_call["strike"]),
                        "dte": int(short_call["entry_dte"]),
                        "predicted_regime": str(short_call["predicted_regime"]),
                        "brace_for_impact_score": float(short_call["brace_for_impact_score"]),
                        "leap_exposure_pct": float(short_call["leap_exposure_pct"]),
                        "overwrite_action": str(short_call["overwrite_action"]),
                        "target_premium_mid": float(short_call["target_premium_mid"]),
                        "vix": float(short_call["vix"]),
                        "vvix": float(short_call["vvix"]),
                        "geometry_stress_score": float(short_call["geometry_stress_score"]),
                        "path_stress_score": float(short_call.get("path_stress_score", 0.0)),
                    }
                )
                if model_name not in {"no_overwrite_baseline", "always_overwrite_baseline"}:
                    safer_reference_pnl = 0.0 if str(config.safer_reference_mode).lower() == "no_overwrite" else option_pnl_dollars
                    dollar_missed_risk = float(option_pnl_dollars - safer_reference_pnl)
                    if dollar_missed_risk < 0:
                        missed_risk.append(
                            {
                                "model_name": model_name,
                                "entry_date": str(short_call["entry_date"]),
                                "exit_date": as_of_text,
                                "predicted_regime": str(short_call["predicted_regime"]),
                                "brace_for_impact_score": float(short_call["brace_for_impact_score"]),
                                "total_trade_pnl": float(option_pnl_dollars),
                                "safer_reference_trade_pnl": float(safer_reference_pnl),
                                "dollar_missed_risk": float(dollar_missed_risk),
                            }
                        )
                short_call = None
                short_call_open = False
                short_call_strike = None
                short_call_entry_premium = None
                short_call_mtm_value = None

        # 3) Recommendation for potential new short call and exposure update
        prediction: dict[str, Any] | None = None
        if model_name not in {"no_overwrite_baseline", "always_overwrite_baseline", "heuristic_policy"}:
            full_index = full_source.index[full_source["date"] == as_of].tolist()
            if full_index:
                idx = int(full_index[0])
                if (idx + 1) >= int(config.min_train_rows):
                    train_start = max(0, idx - int(config.train_lookback_days) + 1)
                    training = full_source.iloc[train_start : idx + 1]
                    prediction = generate_prediction_record(
                        context=create_replay_context(as_of_text),
                        model_name=model_name,
                        train_df=training,
                        min_train_rows=int(config.min_train_rows),
                        n_components=4,
                        random_state=42,
                        covariance_type="diag",
                        precomputed_path_aware_cache=None,
                    )
        recommendation = _recommendation_for_row(
            model_name=model_name,
            row=pd.Series(row._asdict()),
            prediction=prediction,
            config=config,
        )
        pending_exposure = float(recommendation["leap_exposure_pct"]) if leap_open else 0.0

        can_short_call = leap_open or bool(config.allow_naked_short_calls)
        if (
            short_call is None
            and recommendation["overwrite_action"] != "none"
            and recommendation["recommended_dte"] > 0
        ):
            if not can_short_call:
                pass
            else:
                iv = max(vix / 100.0 * MODEL_OPTION_IV_MULTIPLIER, 0.01)
                entry_premium = black_scholes_call_price(
                    spot=spot,
                    strike=float(recommendation["selected_strike"]),
                    dte=int(recommendation["recommended_dte"]),
                    iv_annual=iv,
                    risk_free_rate=config.risk_free_rate,
                    dividend_yield=config.dividend_yield,
                )
                short_call = {
                    "entry_date": as_of_text,
                    "entry_dte": int(recommendation["recommended_dte"]),
                    "days_left": int(recommendation["recommended_dte"]),
                    "strike": float(recommendation["selected_strike"]),
                    "entry_premium": float(entry_premium),
                    "last_option_value": float(entry_premium),
                    "profit_exit_pct": float(recommendation["profit_exit_pct"]),
                    "loss_exit_multiple": float(recommendation["loss_exit_multiple"]),
                    "exit_on_underlying_touch": bool(recommendation["exit_on_underlying_touch"]),
                    "predicted_regime": str(recommendation["predicted_regime"]),
                    "brace_for_impact_score": float(recommendation["brace_for_impact_score"]),
                    "leap_exposure_pct": float(recommendation["leap_exposure_pct"]),
                    "overwrite_action": str(recommendation["overwrite_action"]),
                    "target_premium_mid": float(recommendation["target_premium_mid"]),
                    "vix": vix,
                    "vvix": vvix,
                    "geometry_stress_score": geometry_stress,
                    "path_stress_score": float(
                        dict((prediction or {}).get("model_diagnostics", {})).get("path_features", {}).get(
                            "path_stress_score", 0.0
                        )
                    ),
                }
                short_call_open = True
                short_call_strike = float(short_call["strike"])
                short_call_entry_premium = float(short_call["entry_premium"])
                short_call_mtm_value = float(short_call["last_option_value"])

        # 4) Daily row
        short_call_cumulative_pnl += float(short_call_daily_pnl)
        total_daily = float(leap_daily_pnl + short_call_daily_pnl)
        total_cumulative_pnl += total_daily
        daily_rows.append(
            {
                "model_name": model_name,
                "date": as_of_text,
                "spy_close": float(spot),
                "vix": float(vix),
                "leap_open": bool(leap_open),
                "leap_premium_estimate": float(leap_premium_estimate),
                "leap_delta": float(config.leap_delta),
                "leap_exposure_pct": float(current_exposure),
                "leap_daily_pnl": float(leap_daily_pnl),
                "leap_cumulative_pnl": float(leap_cumulative_pnl),
                "short_call_open": bool(short_call_open),
                "short_call_strike": short_call_strike,
                "short_call_entry_premium": short_call_entry_premium,
                "short_call_mtm_value": short_call_mtm_value,
                "short_call_daily_pnl": float(short_call_daily_pnl),
                "short_call_cumulative_pnl": float(short_call_cumulative_pnl),
                "total_daily_pnl": float(total_daily),
                "total_cumulative_pnl": float(total_cumulative_pnl),
                "overwrite_daily_pnl": float(short_call_daily_pnl),
                "avoided_leap_drawdown_estimate": float(
                    max(0.0, (1.0 - current_exposure) * max((previous_close or spot) - spot, 0.0) * 100.0)
                ),
                "missed_leap_upside_estimate": float(
                    max(0.0, (1.0 - current_exposure) * max(spot - (previous_close or spot), 0.0) * 100.0)
                ),
            }
        )

        current_exposure = pending_exposure
        previous_close = spot

    return trades, daily_rows, missed_risk


def run_policy_backtest(
    *,
    config: PolicyBacktestConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    models: list[str] | None = None,
    run_mode: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source = load_feature_store(config.feature_store_path)
    effective_mode = str(run_mode or config.run_mode or "tuning").strip().lower()
    effective_start, effective_end = _effective_dates(
        source,
        start_date or config.start_date,
        end_date or config.end_date,
        effective_mode,
    )
    scoped = filter_date_range(source, start_date=effective_start, end_date=effective_end)
    if scoped.empty:
        raise RuntimeError("Policy backtester date range has no rows in the feature store.")
    if effective_mode == "tuning" and len(scoped) > TUNING_MAX_DATES:
        scoped = scoped.tail(TUNING_MAX_DATES).reset_index(drop=True)

    selected_models = _resolve_models(models or config.models)
    trades_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for name in selected_models:
        model_trades, model_daily, model_missed = _simulate_one_model(
            model_name=name,
            scoped=scoped,
            full_source=source,
            config=config,
        )
        trades_rows.extend(model_trades)
        daily_rows.extend(model_daily)
        missed_rows.extend(model_missed)

    trades_df = pd.DataFrame(trades_rows)
    daily_df = pd.DataFrame(daily_rows)
    missed_df = pd.DataFrame(missed_rows)

    trades_df = _round_columns(
        trades_df,
        [
            "entry_premium",
            "exit_premium",
            "dollar_pnl",
            "strike",
            "target_premium_mid",
            "vix",
            "vvix",
            "geometry_stress_score",
            "path_stress_score",
        ],
    )
    daily_df = _round_columns(
        daily_df,
        [
            "spy_close",
            "leap_premium_estimate",
            "leap_delta",
            "leap_exposure_pct",
            "leap_daily_pnl",
            "leap_cumulative_pnl",
            "short_call_strike",
            "short_call_entry_premium",
            "short_call_mtm_value",
            "short_call_daily_pnl",
            "short_call_cumulative_pnl",
            "total_daily_pnl",
            "total_cumulative_pnl",
            "overwrite_daily_pnl",
            "avoided_leap_drawdown_estimate",
            "missed_leap_upside_estimate",
        ],
    )
    missed_df = _round_columns(
        missed_df,
        ["total_trade_pnl", "safer_reference_trade_pnl", "dollar_missed_risk"],
    )

    model_summary_df = build_policy_model_summary(trades_df=trades_df, daily_df=daily_df, missed_risk_df=missed_df)
    model_summary_df = _round_columns(
        model_summary_df,
        [
            "total_pnl",
            "leap_pnl",
            "overwrite_pnl",
            "max_drawdown",
            "worst_day_pnl",
            "worst_trade_pnl",
            "average_daily_pnl",
            "median_daily_pnl",
            "volatility_of_daily_pnl",
            "average_overwrite_pnl",
            "median_overwrite_pnl",
            "premium_collected",
            "premium_retained_pct",
            "average_premium_collected",
            "average_premium_lost_on_loss_exit",
            "average_holding_days",
            "tail_loss_1pct",
            "tail_loss_5pct",
            "average_loss_when_model_missed_risk",
            "dollar_missed_risk_score",
            "leap_exposure_avg",
            "avoided_leap_drawdown_estimate",
            "missed_leap_upside_estimate",
            "sharpe_like_score",
            "total_return_pct",
        ],
    )
    exit_summary_df = (
        trades_df.groupby(["model_name", "instrument_type", "exit_reason"], as_index=False)
        .agg(trade_count=("exit_reason", "size"), dollar_pnl=("dollar_pnl", "sum"))
        .sort_values(["model_name", "instrument_type", "trade_count"], ascending=[True, True, False])
        if not trades_df.empty
        else pd.DataFrame(columns=["model_name", "instrument_type", "exit_reason", "trade_count", "dollar_pnl"])
    )
    exit_summary_df = _round_columns(exit_summary_df, ["dollar_pnl"])
    worst_trades_df = trades_df.sort_values("dollar_pnl", ascending=True).head(20) if not trades_df.empty else pd.DataFrame()
    best_trades_df = trades_df.sort_values("dollar_pnl", ascending=False).head(20) if not trades_df.empty else pd.DataFrame()
    starting_assumptions_df = _build_model_starting_assumptions(
        model_names=selected_models,
        scoped=scoped,
        config=config,
        start_date=effective_start,
        end_date=effective_end,
    )
    audit_daily_df = _build_policy_mechanics_audit_daily(daily_df)
    audit_trades_df = _build_policy_mechanics_audit_trades(trades_df)
    invariant_checks_df = _build_policy_invariant_checks(
        model_summary_df=model_summary_df,
        trades_df=trades_df,
        daily_df=daily_df,
        config=config,
    )
    profit_loss_explanation_df = _build_policy_profit_loss_explanation(
        model_summary_df=model_summary_df,
        trades_df=trades_df,
        daily_df=daily_df,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_trades_path = output_dir / "policy_trades.csv"
    policy_daily_path = output_dir / "policy_daily_pnl.csv"
    policy_summary_path = output_dir / "policy_model_summary.csv"
    policy_exit_path = output_dir / "policy_exit_summary.csv"
    policy_worst_path = output_dir / "policy_worst_trades.csv"
    policy_missed_path = output_dir / "policy_dollar_missed_risk.csv"
    policy_audit_assumptions_path = output_dir / "policy_audit_starting_assumptions.csv"
    policy_audit_daily_path = output_dir / "policy_audit_first_20_daily_rows.csv"
    policy_audit_trades_path = output_dir / "policy_audit_first_20_trades.csv"
    policy_invariant_checks_path = output_dir / "policy_invariant_checks.csv"
    policy_profit_loss_explanation_path = output_dir / "policy_profit_loss_explanation.csv"
    report_path = output_dir / "policy_backtest_report.md"

    trades_df.to_csv(policy_trades_path, index=False)
    daily_df.to_csv(policy_daily_path, index=False)
    model_summary_df.to_csv(policy_summary_path, index=False)
    exit_summary_df.to_csv(policy_exit_path, index=False)
    worst_trades_df.to_csv(policy_worst_path, index=False)
    missed_df.to_csv(policy_missed_path, index=False)
    starting_assumptions_df.to_csv(policy_audit_assumptions_path, index=False)
    audit_daily_df.to_csv(policy_audit_daily_path, index=False)
    audit_trades_df.to_csv(policy_audit_trades_path, index=False)
    invariant_checks_df.to_csv(policy_invariant_checks_path, index=False)
    profit_loss_explanation_df.to_csv(policy_profit_loss_explanation_path, index=False)

    report_lines = [
        "# Economic Policy Backtest",
        "",
        "Guardrail note: This is a V1 economic approximation using model-estimated option prices (Black-Scholes proxy),",
        "not exact historical option-chain fills.",
        "",
        "## Model Economic Leaderboard",
        "",
        (model_summary_df.to_markdown(index=False) if not model_summary_df.empty else "_No rows_"),
        "",
        "## LEAP PnL vs Overwrite PnL Decomposition",
        "",
        (
            model_summary_df[["model_name", "leap_pnl", "overwrite_pnl", "total_pnl"]].to_markdown(index=False)
            if not model_summary_df.empty
            else "_No rows_"
        ),
        "",
        "## Worst 20 Trades",
        "",
        (worst_trades_df.to_markdown(index=False) if not worst_trades_df.empty else "_No rows_"),
        "",
        "## Best 20 Trades",
        "",
        (best_trades_df.to_markdown(index=False) if not best_trades_df.empty else "_No rows_"),
        "",
        "## Missed Risk Dollar Impact",
        "",
        (missed_df.head(20).to_markdown(index=False) if not missed_df.empty else "_No rows_"),
        "",
        "## Exit Reason Summary",
        "",
        (exit_summary_df.to_markdown(index=False) if not exit_summary_df.empty else "_No rows_"),
        "",
        "## Policy Mechanics Audit",
        "",
        "### Starting Assumptions",
        "",
        (starting_assumptions_df.to_markdown(index=False) if not starting_assumptions_df.empty else "_No rows_"),
        "",
        "### First 20 Daily Rows Per Model",
        "",
        (audit_daily_df.to_markdown(index=False) if not audit_daily_df.empty else "_No rows_"),
        "",
        "### First 20 Trades Per Model",
        "",
        (audit_trades_df.to_markdown(index=False) if not audit_trades_df.empty else "_No rows_"),
        "",
        "### Invariant Checks",
        "",
        (invariant_checks_df.to_markdown(index=False) if not invariant_checks_df.empty else "_No rows_"),
        "",
        "### Why This Model Made or Lost Money",
        "",
        (profit_loss_explanation_df.to_markdown(index=False) if not profit_loss_explanation_df.empty else "_No rows_"),
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    elapsed = time.perf_counter() - started
    runtime_profile = {
        "total_seconds": round(float(elapsed), 6),
        "rows_replayed": int(len(scoped)),
        "models_count": int(len(selected_models)),
    }
    (output_dir / "runtime_profile.json").write_text(json.dumps(runtime_profile, indent=2), encoding="utf-8")

    preview_cap = 50 if effective_mode == "tuning" else 100
    return {
        "run_mode": effective_mode,
        "date_start": effective_start,
        "date_end": effective_end,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "policy_trades_path": str(policy_trades_path),
        "policy_daily_pnl_path": str(policy_daily_path),
        "policy_model_summary_path": str(policy_summary_path),
        "policy_exit_summary_path": str(policy_exit_path),
        "policy_worst_trades_path": str(policy_worst_path),
        "policy_dollar_missed_risk_path": str(policy_missed_path),
        "policy_audit_starting_assumptions_path": str(policy_audit_assumptions_path),
        "policy_audit_first_20_daily_rows_path": str(policy_audit_daily_path),
        "policy_audit_first_20_trades_path": str(policy_audit_trades_path),
        "policy_invariant_checks_path": str(policy_invariant_checks_path),
        "policy_profit_loss_explanation_path": str(policy_profit_loss_explanation_path),
        "runtime_profile": runtime_profile,
        "model_economic_leaderboard": model_summary_df.head(preview_cap).to_dict(orient="records"),
        "worst_trades": worst_trades_df.head(10).to_dict(orient="records"),
        "best_trades": best_trades_df.head(10).to_dict(orient="records"),
        "policy_invariant_checks": invariant_checks_df.head(preview_cap).to_dict(orient="records"),
        "policy_profit_loss_explanation": profit_loss_explanation_df.head(preview_cap).to_dict(orient="records"),
        "policy_audit_daily_preview": audit_daily_df.head(preview_cap).to_dict(orient="records"),
        "policy_audit_trades_preview": audit_trades_df.head(preview_cap).to_dict(orient="records"),
    }
