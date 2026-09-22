from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import pytest

from market_physics_core.adapters.baseline_store.local import write_local_history
from market_physics_core.applications.skew_baseline.service import (
    _trading_session_dte,
)
from market_physics_core.core.statistics import describe
from market_physics_core.core.volatility import (
    interpolate_iv,
    vix_scaled_strike_targets,
)


def test_horizon_targets_and_supported_interpolation():
    one_day = vix_scaled_strike_targets(760.0, 15.87, 1)
    two_day = vix_scaled_strike_targets(760.0, 15.87, 2)
    assert one_day.absolute_distance == pytest.approx(3.8, abs=0.01)
    assert two_day.absolute_distance == pytest.approx(5.37, abs=0.01)
    result = interpolate_iv([(755.0, 0.20), (757.0, 0.18)], 756.0)
    assert result.value == pytest.approx(0.19)
    with pytest.raises(ValueError, match="outside"):
        interpolate_iv([(755.0, 0.20), (757.0, 0.18)], 758.0)


def test_xnys_trading_session_dte_handles_weekends():
    calendar = xcals.get_calendar("XNYS")
    assert _trading_session_dte(
        calendar, date(2026, 9, 17), date(2026, 9, 18)
    ) == 1
    assert _trading_session_dte(
        calendar, date(2026, 9, 18), date(2026, 9, 21)
    ) == 1


def test_describe_and_immutable_local_write(tmp_path: Path):
    assert describe([1.0, 2.0, 3.0])["sample_standard_deviation"] == 1.0
    summary = {
        "metric_id": "spy_iv_skew_vix_scaled",
        "sigma_multiplier": 0.5,
        "methodology_version": "v2",
        "as_of_session": "2026-09-17",
        "run_id": "run-test",
        "status": "partial",
        "window_start": "2026-08-13",
        "window_end": "2026-09-17",
        "dte_groups": [],
    }
    run_dir, checksums = write_local_history(
        tmp_path, summary, [{"session": "2026-09-17", "eligible": False}]
    )
    assert set(checksums) == {"summary.json", "daily_skew.csv", "report.md"}
    assert run_dir.parts[-4:] == ("sigma_0p5", "v2", "2026-09-17", "run-test")
