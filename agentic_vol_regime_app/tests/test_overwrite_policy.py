from __future__ import annotations

from src.backtest.policy.overwrite_policy import (
    candidate_short_call_strike,
    expected_upside_move_pct,
    nearest_listed_strike,
    vix_premium_target,
)


def test_vix_premium_target_ranges() -> None:
    low = vix_premium_target(18.0)
    mid = vix_premium_target(24.0)
    high = vix_premium_target(36.0)
    assert low.low == 1.20 and low.high == 1.50
    assert mid.low == 1.50 and mid.high == 1.80
    assert high.low == 2.00 and high.high == 3.00


def test_expected_move_and_strike_rounding() -> None:
    pct = expected_upside_move_pct(16.0)
    assert pct > 0.0
    strike = candidate_short_call_strike(700.0, 16.0)
    assert strike > 700.0
    assert nearest_listed_strike(strike, increment=1.0) == round(strike)

