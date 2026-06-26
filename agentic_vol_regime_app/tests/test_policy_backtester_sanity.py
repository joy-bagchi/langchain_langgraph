from __future__ import annotations


def test_sanity_leap_delta_pnl_formula() -> None:
    leap_delta = 0.70
    multiplier = 100
    exposure = 1.0
    spot_change = 10.0
    pnl = leap_delta * multiplier * exposure * spot_change
    assert pnl == 700.0


def test_sanity_short_call_dollar_pnl_formula() -> None:
    multiplier = 100
    entry = 1.50
    profit_exit = 0.30
    loss_exit = 3.00
    pnl_profit = (entry - profit_exit) * multiplier
    pnl_loss = (entry - loss_exit) * multiplier
    assert pnl_profit == 120.0
    assert pnl_loss == -150.0


def test_sanity_combined_portfolio_arithmetic() -> None:
    leap = 700.0
    short_call = -150.0
    total = leap + short_call
    assert total == 550.0


def test_sanity_leap_premium_thresholds() -> None:
    entry_premium = 100.0
    cumulative_pnl_profit = 2000.0
    cumulative_pnl_loss = -2000.0
    premium_profit = entry_premium + (cumulative_pnl_profit / 100.0)
    premium_loss = entry_premium + (cumulative_pnl_loss / 100.0)
    assert premium_profit == 120.0
    assert premium_loss == 80.0
