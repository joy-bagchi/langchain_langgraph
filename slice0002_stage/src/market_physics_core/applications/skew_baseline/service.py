"""SPY VIX-scaled skew-baseline application."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from hashlib import sha256
import json
import math
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_physics_core.adapters.market_data.gcs import MarketInputs
from market_physics_core.core.statistics import describe
from market_physics_core.core.volatility import (
    interpolate_iv,
    vix_scaled_strike_targets,
)


def _trading_session_dte(calendar: Any, observation: date, expiry: date) -> int:
    if expiry <= observation:
        return 0
    sessions = calendar.sessions_in_range(
        pd.Timestamp(observation + timedelta(days=1)), pd.Timestamp(expiry)
    )
    return len(sessions)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def calculate_baseline(
    inputs: MarketInputs,
    *,
    code_commit: str,
    sigma_multiplier: float = 0.5,
    methodology_version: str = "v2",
    rolling_window_sessions: int = 26,
    calendar_name: str = "XNYS",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not inputs.surfaces:
        raise ValueError("option catalog contains no surface snapshots")
    calendar = xcals.get_calendar(calendar_name)
    as_of = pd.Timestamp(inputs.surfaces[-1].observation_date)
    as_of_session = calendar.date_to_session(as_of, direction="none")
    window = calendar.sessions_window(as_of_session, -(rolling_window_sessions - 1))
    window_start = window[0].date().isoformat()
    window_end = window[-1].date().isoformat()
    surfaces = [
        item
        for item in inputs.surfaces
        if window_start <= item.observation_date <= window_end
    ]

    history = inputs.history.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.date
    history = history.set_index("date", verify_integrity=True)
    observations: list[dict[str, Any]] = []
    history_source = next(
        source
        for source in inputs.source_objects
        if "vol-regime-history" in source.object
        and source.object.endswith(".parquet")
    )

    for snapshot in surfaces:
        session = date.fromisoformat(snapshot.observation_date)
        frame = snapshot.frame
        base = {
            "session": snapshot.observation_date,
            "market_timezone": "America/New_York",
            "capture_timestamp": snapshot.observation_time,
            "input_representation": "raw_put_call_observations",
            "option_source_uri": snapshot.source_object.uri,
            "option_source_generation": snapshot.source_object.generation,
            "option_source_sha256": snapshot.source_object.sha256,
            "history_source_uri": history_source.uri,
            "history_source_generation": history_source.generation,
            "history_source_sha256": history_source.sha256,
        }
        for target_dte in (1, 2):
            row: dict[str, Any] = {**base, "trading_session_dte": target_dte}
            if session not in history.index:
                row.update(eligible=False, exclusion_reason="missing_spy_vix_history")
                observations.append(row)
                continue
            history_row = history.loc[session]
            spy_close, vix = float(history_row["SPY"]), float(history_row["VIX"])
            if not (math.isfinite(spy_close) and math.isfinite(vix)):
                row.update(eligible=False, exclusion_reason="missing_spy_or_vix_value")
                observations.append(row)
                continue
            snapshot_spots = pd.to_numeric(
                frame["underlying_price"], errors="coerce"
            ).dropna().unique()
            if len(snapshot_spots) != 1:
                row.update(
                    eligible=False,
                    exclusion_reason="invalid_snapshot_spot_cardinality",
                )
                observations.append(row)
                continue
            spot = float(snapshot_spots[0])
            targets = vix_scaled_strike_targets(
                spot, vix, target_dte, sigma_multiplier
            )
            vix_timestamp = calendar.session_close(
                pd.Timestamp(snapshot.observation_date)
            ).isoformat()
            row.update(
                spot=spot,
                spot_source="option_surface_snapshot_underlying_price",
                spy_close=spy_close,
                spy_close_source="IBKR_daily_TRADES",
                spy_close_minus_snapshot_spot=spy_close - spot,
                vix=vix,
                vix_timestamp=vix_timestamp,
                vix_timestamp_basis="XNYS_session_close_for_daily_bar",
                vix_source="IBKR_daily_TRADES",
                horizon_vol_fraction=targets.horizon_vol_fraction,
                sigma_multiplier=sigma_multiplier,
                distance_fraction=targets.distance_fraction,
                absolute_strike_distance=targets.absolute_distance,
                put_target=targets.put_target,
                call_target=targets.call_target,
            )
            expiry_candidates: list[tuple[date, int]] = []
            for expiry_value in sorted(frame["expiry"].astype(str).unique()):
                expiry = date.fromisoformat(expiry_value)
                if _trading_session_dte(calendar, session, expiry) == target_dte:
                    calendar_dte = (expiry - session).days
                    expiry_candidates.append((expiry, calendar_dte))
            if not expiry_candidates:
                row.update(
                    eligible=False,
                    exclusion_reason="missing_supported_expiration",
                )
                observations.append(row)
                continue
            expiry, calendar_dte = expiry_candidates[0]
            expiry_frame = frame[
                frame["expiry"].astype(str).eq(expiry.isoformat())
            ]
            try:
                put_rows = expiry_frame[
                    expiry_frame["right"].astype(str).str.upper().eq("P")
                ]
                call_rows = expiry_frame[
                    expiry_frame["right"].astype(str).str.upper().eq("C")
                ]
                put_iv = interpolate_iv(
                    zip(put_rows["strike"], put_rows["implied_vol"]),
                    targets.put_target,
                )
                call_iv = interpolate_iv(
                    zip(call_rows["strike"], call_rows["implied_vol"]),
                    targets.call_target,
                )
            except ValueError as exc:
                row.update(
                    eligible=False,
                    exclusion_reason=f"interpolation_quality_failure: {exc}",
                    expiration=expiry.isoformat(),
                    calendar_dte=calendar_dte,
                )
                observations.append(row)
                continue
            row.update(
                eligible=True,
                exclusion_reason="",
                expiration=expiry.isoformat(),
                calendar_dte=calendar_dte,
                put_iv=put_iv.value,
                call_iv=call_iv.value,
                put_lower_strike=put_iv.lower_strike,
                put_upper_strike=put_iv.upper_strike,
                call_lower_strike=call_iv.lower_strike,
                call_upper_strike=call_iv.upper_strike,
                skew_iv_percentage_points=(put_iv.value - call_iv.value) * 100.0,
            )
            observations.append(row)

    config_identity = {
        "instrument": "SPY",
        "metric_id": "spy_iv_skew_vix_scaled",
        "methodology_version": methodology_version,
        "sigma_multiplier": sigma_multiplier,
        "annualization_trading_sessions": 252,
        "calendar": calendar_name,
        "rolling_window_sessions": rolling_window_sessions,
        "snapshot_selection_policy": "catalog_selected_single_post_close_snapshot",
        "spot_source": "option_surface_snapshot_underlying_price",
        "vix_policy": "same_session_daily_close_for_verified_closing_surface",
        "interpolation_policy": "linear_within_same_expiry_and_right_no_extrapolation",
    }
    configuration_hash = _canonical_hash(config_identity)
    source_identity = [source.as_dict() for source in inputs.source_objects]
    idempotency_key = _canonical_hash(
        {
            "configuration_hash": configuration_hash,
            "code_commit": code_commit,
            "as_of_session": window_end,
            "source_objects": source_identity,
        }
    )

    dte_groups: list[dict[str, Any]] = []
    week = as_of.date().isocalendar()
    week_id = f"{week.year}-W{week.week:02d}"
    for target_dte in (1, 2):
        rows = [
            row for row in observations if row["trading_session_dte"] == target_dte
        ]
        usable = [row for row in rows if row.get("eligible")]
        values = [row["skew_iv_percentage_points"] for row in usable]
        exclusions = Counter(
            row["exclusion_reason"] for row in rows if not row.get("eligible")
        )
        latest = usable[-1] if usable else None
        prior_values = (
            [row["skew_iv_percentage_points"] for row in usable[:-1]]
            if latest
            else []
        )
        weekly_values = [
            row["skew_iv_percentage_points"]
            for row in usable
            if date.fromisoformat(row["session"]).isocalendar()[:2]
            == (week.year, week.week)
        ]
        dte_groups.append(
            {
                "trading_session_dte": target_dte,
                "dte_convention": "XNYS_trading_sessions_after_capture_through_expiry",
                "window_start": window_start,
                "window_end": window_end,
                "nominal_session_count": rolling_window_sessions,
                "available_surface_dates": len(rows),
                "usable_n": len(usable),
                "coverage": (
                    "complete"
                    if len(usable) == rolling_window_sessions
                    else "partial"
                ),
                "exclusions": dict(sorted(exclusions.items())),
                "statistics": describe(values),
                "weekly_statistics": describe(weekly_values),
                "prior_comparable_week": None,
                "prior_comparable_differences": None,
                "latest": (
                    {
                        "session": latest["session"],
                        "skew_iv_percentage_points": latest[
                            "skew_iv_percentage_points"
                        ],
                        "prior_days_statistics": describe(prior_values),
                        "difference_from_prior_mean": (
                            latest["skew_iv_percentage_points"]
                            - float(describe(prior_values)["mean"])
                            if prior_values
                            else None
                        ),
                    }
                    if latest
                    else None
                ),
            }
        )
    total_usable = sum(group["usable_n"] for group in dte_groups)
    status = (
        "insufficient_data"
        if total_usable == 0
        else (
            "complete"
            if all(group["coverage"] == "complete" for group in dte_groups)
            else "partial"
        )
    )
    summary = {
        "schema_version": "1",
        "methodology_version": methodology_version,
        "instrument": "SPY",
        "metric_id": "spy_iv_skew_vix_scaled",
        "code_commit": code_commit,
        "configuration_hash": configuration_hash,
        "idempotency_key": idempotency_key,
        "run_id": f"run-{idempotency_key[:16]}",
        "as_of_session": window_end,
        "market_timezone": "America/New_York",
        "week_id": week_id,
        "window_start": window_start,
        "window_end": window_end,
        "nominal_session_count": rolling_window_sessions,
        "sigma_multiplier": sigma_multiplier,
        "annualization_trading_sessions": 252,
        "strike_distance_policy": "vix_horizon_scaled_half_volatility",
        "snapshot_selection_policy": config_identity["snapshot_selection_policy"],
        "spot_source": config_identity["spot_source"],
        "vix_alignment_policy": config_identity["vix_policy"],
        "iv_units": "percentage_points",
        "interpolation_quality_policy": config_identity["interpolation_policy"],
        "source_objects": source_identity,
        "source_history_warnings": inputs.history_metadata.get("warnings", []),
        "dte_groups": dte_groups,
        "status": status,
    }
    return summary, observations


__all__ = ["calculate_baseline"]
