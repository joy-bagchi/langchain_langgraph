import pandas as pd

from vol_visualizer.cube import build_iv_cube


def test_build_iv_cube_filters_by_option_right() -> None:
    frame = pd.DataFrame([
        {"observation_time": "2026-08-07T20:15:00Z", "observation_date": "2026-08-07", "symbol": "SPY", "expiry": "2026-08-14", "strike": 640.0, "dte": 7, "right": "C", "implied_vol": 0.2},
        {"observation_time": "2026-08-07T20:15:00Z", "observation_date": "2026-08-07", "symbol": "SPY", "expiry": "2026-08-14", "strike": 640.0, "dte": 7, "right": "P", "implied_vol": 0.21},
    ])
    assert build_iv_cube(frame, right="C").right.tolist() == ["C"]
