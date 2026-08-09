from datetime import date, datetime
from zoneinfo import ZoneInfo

from vol_surface_publisher.contracts import is_us_equity_market_session, latest_completed_option_session
from vol_surface_publisher.publisher import collect_and_publish


def test_us_equity_market_sessions_exclude_weekends_and_recurring_holidays() -> None:
    assert not is_us_equity_market_session(date(2026, 8, 8))  # Saturday
    assert not is_us_equity_market_session(date(2026, 7, 3))  # Independence Day observed
    assert not is_us_equity_market_session(date(2026, 11, 26))  # Thanksgiving
    assert is_us_equity_market_session(date(2026, 8, 10))


def test_latest_completed_session_uses_frozen_data_after_close_and_on_weekends() -> None:
    ny = ZoneInfo("America/New_York")
    assert latest_completed_option_session(datetime(2026, 8, 7, 16, 15, tzinfo=ny)) == (date(2026, 8, 7), 2)
    assert latest_completed_option_session(datetime(2026, 8, 8, 12, 0, tzinfo=ny)) == (date(2026, 8, 7), 2)
    assert latest_completed_option_session(datetime(2026, 8, 10, 12, 0, tzinfo=ny)) is None


def test_intraday_run_waits_without_opening_ibkr_connection() -> None:
    result = collect_and_publish(now=datetime(2026, 8, 10, 12, 0, tzinfo=ZoneInfo("America/New_York")))
    assert result["status"] == "skipped_waiting_for_market_close"
