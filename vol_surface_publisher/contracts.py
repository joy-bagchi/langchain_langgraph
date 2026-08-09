"""Stable tabular contract for dated option implied-volatility observations."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

OPTION_CHAIN_SCHEMA_VERSION = "option_chain_iv.v1"
MANIFEST_SCHEMA_VERSION = "option_chain_iv_catalog.v1"
REQUIRED_COLUMNS = ("observation_time", "observation_date", "symbol", "expiry", "strike", "dte", "right", "implied_vol")
NEW_YORK = ZoneInfo("America/New_York")
EOD_CAPTURE_TIME = time(hour=16, minute=15)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Gregorian computus, sufficient for the NYSE Good Friday closure."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day_of_month = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day_of_month + 1)


def is_us_equity_market_session(day: date) -> bool:
    """Return whether a normal NYSE session is scheduled for *day*.

    This intentionally covers recurring full-day closures. Exceptional one-off
    exchange closures are still caught by the later IBKR-data diagnostic.
    """
    if day.weekday() >= 5:
        return False
    holidays = {
        _observed(date(day.year, 1, 1)),
        _nth_weekday(day.year, 1, 0, 3),
        _nth_weekday(day.year, 2, 0, 3),
        _easter_sunday(day.year) - timedelta(days=2),
        _last_weekday(day.year, 5, 0),
        _observed(date(day.year, 6, 19)) if day.year >= 2022 else date.min,
        _observed(date(day.year, 7, 4)),
        _nth_weekday(day.year, 9, 0, 1),
        _nth_weekday(day.year, 11, 3, 4),
        _observed(date(day.year, 12, 25)),
    }
    # New Year's Day may be observed in the prior calendar year.
    holidays.add(_observed(date(day.year + 1, 1, 1)))
    return day not in holidays


def latest_completed_option_session(now: datetime | None = None) -> tuple[date, int] | None:
    """Return the latest completed U.S. session and IBKR market-data type.

    The publisher intentionally avoids an intraday surface. Once the session
    is complete it requests frozen data (IBKR type 2), which is the last quote
    recorded at the close. Before the close there is no EOD observation yet.
    """
    localized = (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)
    today = localized.date()
    if is_us_equity_market_session(today) and localized.timetz().replace(tzinfo=None) >= EOD_CAPTURE_TIME:
        return today, 2
    if is_us_equity_market_session(today):
        return None
    cursor = today
    while not is_us_equity_market_session(cursor):
        cursor -= timedelta(days=1)
    return cursor, 2


def option_chain_frame(snapshot: dict[str, Any], *, observation_time: datetime | None = None) -> pd.DataFrame:
    """Extract valid IV quotes from one normalized IBKR option-chain snapshot."""
    observed_at = pd.Timestamp(observation_time or snapshot["as_of"])
    observed_at = observed_at.tz_localize("UTC") if observed_at.tzinfo is None else observed_at.tz_convert("UTC")
    chain = dict(snapshot.get("option_chain", {}))
    symbol = str(chain.get("underlying_symbol") or next(iter(snapshot.get("symbols", {})), "")).upper()
    rows: list[dict[str, Any]] = []
    for quote in chain.get("option_quotes", []):
        try:
            expiry_day = pd.Timestamp(str(quote["expiry"])).date()
            strike, iv = float(quote["strike"]), float(dict(quote.get("greeks") or {})["implied_vol"])
        except (KeyError, TypeError, ValueError):
            continue
        dte = (expiry_day - observed_at.date()).days
        if iv <= 0 or dte < 0:
            continue
        rows.append({"observation_time": observed_at.isoformat(), "observation_date": observed_at.date().isoformat(),
                     "symbol": symbol, "expiry": expiry_day.isoformat(), "strike": strike, "dte": dte,
                     "right": str(quote.get("right", "")).upper(), "implied_vol": iv, "bid": quote.get("bid"),
                     "ask": quote.get("ask"), "mark": quote.get("mark"), "underlying_price": chain.get("underlying_price"),
                     "source": str(snapshot.get("source", "IBKR"))})
    return pd.DataFrame(rows, columns=[*REQUIRED_COLUMNS, "bid", "ask", "mark", "underlying_price", "source"])


def validate_option_chain_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError("Option-surface frame must contain non-empty IV observations and all required columns.")
    if frame.implied_vol.isna().any() or (frame.implied_vol <= 0).any() or frame.dte.isna().any() or (frame.dte < 0).any():
        raise ValueError("Option-surface implied_vol must be positive and dte must be non-negative.")
