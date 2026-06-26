from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(slots=True)
class PremiumTarget:
    low: float
    high: float
    mid: float


def vix_premium_target(vix: float) -> PremiumTarget:
    value = float(vix)
    if value <= 20.0:
        low, high = 1.20, 1.50
    elif value <= 30.0:
        low, high = 1.50, 1.80
    else:
        low, high = 2.00, 3.00
    return PremiumTarget(low=low, high=high, mid=(low + high) / 2.0)


def expected_upside_move_pct(vix: float) -> float:
    daily_vol_pct = float(vix) / sqrt(252.0) / 100.0
    return daily_vol_pct / 2.0


def candidate_short_call_strike(spot: float, vix: float) -> float:
    upside_pct = expected_upside_move_pct(vix)
    return float(spot) * (1.0 + upside_pct)


def nearest_listed_strike(raw_strike: float, *, increment: float = 1.0) -> float:
    step = max(float(increment), 0.5)
    return round(round(float(raw_strike) / step) * step, 2)

