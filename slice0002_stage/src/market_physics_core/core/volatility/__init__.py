"""Pure volatility and option-skew calculations."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class StrikeTargets:
    horizon_vol_fraction: float
    distance_fraction: float
    absolute_distance: float
    put_target: float
    call_target: float


@dataclass(frozen=True)
class InterpolatedIV:
    value: float
    lower_strike: float
    upper_strike: float


def vix_scaled_strike_targets(
    spot: float,
    vix: float,
    trading_session_dte: int,
    sigma_multiplier: float = 0.5,
    annualization_sessions: int = 252,
) -> StrikeTargets:
    values = (spot, vix, sigma_multiplier)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("spot, VIX, and sigma multiplier must be positive finite values")
    if trading_session_dte <= 0 or annualization_sessions <= 0:
        raise ValueError("DTE and annualization sessions must be positive")
    horizon = (vix / 100.0) * math.sqrt(
        trading_session_dte / annualization_sessions
    )
    distance_fraction = sigma_multiplier * horizon
    absolute_distance = spot * distance_fraction
    return StrikeTargets(
        horizon_vol_fraction=horizon,
        distance_fraction=distance_fraction,
        absolute_distance=absolute_distance,
        put_target=spot - absolute_distance,
        call_target=spot + absolute_distance,
    )


def interpolate_iv(
    strike_iv_pairs: Iterable[tuple[float, float]], target_strike: float
) -> InterpolatedIV:
    if not math.isfinite(target_strike):
        raise ValueError("target strike must be finite")
    points: dict[float, float] = {}
    for strike, implied_vol in strike_iv_pairs:
        if not (
            math.isfinite(strike)
            and strike > 0
            and math.isfinite(implied_vol)
            and implied_vol > 0
        ):
            continue
        if strike in points and not math.isclose(points[strike], implied_vol):
            raise ValueError(f"conflicting IV values at strike {strike}")
        points[strike] = implied_vol
    strikes = sorted(points)
    position = bisect_left(strikes, target_strike)
    if position < len(strikes) and math.isclose(
        strikes[position], target_strike, rel_tol=0.0, abs_tol=1e-12
    ):
        strike = strikes[position]
        return InterpolatedIV(points[strike], strike, strike)
    if position == 0 or position == len(strikes):
        raise ValueError("target strike is outside the supported strike bracket")
    lower, upper = strikes[position - 1], strikes[position]
    weight = (target_strike - lower) / (upper - lower)
    value = points[lower] + weight * (points[upper] - points[lower])
    return InterpolatedIV(value, lower, upper)


__all__ = [
    "InterpolatedIV",
    "StrikeTargets",
    "interpolate_iv",
    "vix_scaled_strike_targets",
]
