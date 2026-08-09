import pandas as pd
import pytest

from vol_visualizer.cube import _surface_grid, build_iv_cube, select_front_monthly_atm, select_monthly_atm


def test_build_iv_cube_filters_by_option_right() -> None:
    frame = pd.DataFrame([
        {"observation_time": "2026-08-07T20:15:00Z", "observation_date": "2026-08-07", "symbol": "SPY", "expiry": "2026-08-14", "strike": 640.0, "dte": 7, "right": "C", "implied_vol": 0.2},
        {"observation_time": "2026-08-07T20:15:00Z", "observation_date": "2026-08-07", "symbol": "SPY", "expiry": "2026-08-14", "strike": 640.0, "dte": 7, "right": "P", "implied_vol": 0.21},
    ])
    assert build_iv_cube(frame, right="C").right.tolist() == ["C"]


def test_surface_grid_interpolates_interior_missing_quote_cells() -> None:
    frame = pd.DataFrame([
        {"dte": 7, "strike": 630.0, "implied_vol": 0.22},
        {"dte": 7, "strike": 650.0, "implied_vol": 0.20},
        {"dte": 14, "strike": 630.0, "implied_vol": 0.23},
        {"dte": 14, "strike": 640.0, "implied_vol": 0.21},
        {"dte": 14, "strike": 650.0, "implied_vol": 0.19},
    ])
    grid = _surface_grid(frame)
    assert grid.loc[7, 640.0] == pytest.approx(0.21)


def test_select_front_monthly_atm_uses_third_friday_and_nearest_strike() -> None:
    rows = []
    for observed, underlying in (("2026-08-07", 641.0), ("2026-08-10", 649.0)):
        for expiry, dte in (("2026-08-14", 7), ("2026-08-21", 14)):
            for strike in (640.0, 650.0):
                for right in ("C", "P"):
                    rows.append({"observation_time": f"{observed}T20:15:00Z", "observation_date": observed,
                                 "symbol": "SPY", "expiry": expiry, "strike": strike, "dte": dte,
                                 "right": right, "implied_vol": 0.2, "underlying_price": underlying})
    selected = select_front_monthly_atm(pd.DataFrame(rows))
    assert selected.monthly_expiry.unique().tolist() == ["2026-08-21"]
    assert selected.groupby("observation_date").atm_strike.first().to_dict() == {"2026-08-07": 640.0, "2026-08-10": 650.0}


def test_select_monthly_atm_uses_requested_monthly_maturity() -> None:
    rows = []
    for expiry, dte in (("2026-08-21", 14), ("2026-09-18", 42), ("2026-10-16", 70)):
        for strike in (640.0, 650.0):
            for right in ("C", "P"):
                rows.append({"observation_time": "2026-08-07T20:15:00Z", "observation_date": "2026-08-07",
                             "symbol": "SPY", "expiry": expiry, "strike": strike, "dte": dte,
                             "right": right, "implied_vol": 0.2, "underlying_price": 649.0})
    selected = select_monthly_atm(pd.DataFrame(rows), monthly_offset=1)
    assert selected.monthly_expiry.unique().tolist() == ["2026-09-18"]
    assert selected.atm_strike.unique().tolist() == [650.0]
