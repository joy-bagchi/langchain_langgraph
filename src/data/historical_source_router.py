from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from src.data.yahoo_eod_provider import (
    YahooEODFetchResult,
    YahooEODPoint,
    fetch_yahoo_eod_history,
    fetch_yahoo_eod_history_with_metadata,
)


@dataclass(slots=True, frozen=True)
class StitchedHistoryResult:
    values: list[float]
    coverage: dict[str, object]
    warnings: list[str]


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _ibkr_series(values: list[float], *, as_of_date: date) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    index = pd.bdate_range(end=as_of_date, periods=len(values)).date
    return pd.Series([float(item) for item in values], index=index, dtype=float)


def _yahoo_series(points: list[YahooEODPoint]) -> pd.Series:
    if not points:
        return pd.Series(dtype=float)
    by_date: dict[date, float] = {}
    for point in points:
        by_date[point.day] = float(point.close)
    days = sorted(by_date.keys())
    return pd.Series([float(by_date[day]) for day in days], index=days, dtype=float)


def stitch_history_with_yahoo_fallback(
    *,
    symbol: str,
    ibkr_values: list[float],
    as_of_date: str | date,
    preferred_start_date: str | date,
    minimum_required_start_date: str | date,
    yahoo_fetcher: Callable[..., list[YahooEODPoint]] = fetch_yahoo_eod_history,
    yahoo_fetcher_with_metadata: Callable[..., YahooEODFetchResult] = fetch_yahoo_eod_history_with_metadata,
) -> StitchedHistoryResult:
    symbol_upper = str(symbol).strip().upper()
    as_of = _parse_date(as_of_date)
    preferred_start = _parse_date(preferred_start_date)
    minimum_required_start = _parse_date(minimum_required_start_date)

    ibkr = _ibkr_series(list(ibkr_values), as_of_date=as_of)
    ibkr_earliest = str(ibkr.index.min()) if not ibkr.empty else ""
    ibkr_latest = str(ibkr.index.max()) if not ibkr.empty else ""

    yahoo_points: list[YahooEODPoint] = []
    warnings: list[str] = []
    yahoo_provider_used = "none"
    yfinance_adapter_failure = False
    external_provider_available = False
    if ibkr.empty or ibkr.index.min() > preferred_start:
        yahoo_end = (ibkr.index.min() - timedelta(days=1)) if not ibkr.empty else as_of
        try:
            if yahoo_fetcher is not fetch_yahoo_eod_history:
                yahoo_points = list(
                    yahoo_fetcher(
                        symbol=symbol_upper,
                        start_date=preferred_start.isoformat(),
                        end_date=(yahoo_end + timedelta(days=1)).isoformat(),
                    )
                )
                yahoo_provider_used = "custom_fetcher"
                external_provider_available = bool(yahoo_points)
            else:
                result = yahoo_fetcher_with_metadata(
                    symbol=symbol_upper,
                    start_date=preferred_start.isoformat(),
                    end_date=(yahoo_end + timedelta(days=1)).isoformat(),
                )
                yahoo_points = list(result.points)
                yahoo_provider_used = str(result.provider_used)
                yfinance_adapter_failure = bool(result.yfinance_adapter_failure)
                external_provider_available = bool(result.external_provider_available)
                warnings.extend(list(result.warnings))
        except Exception as exc:
            warnings.append(f"Yahoo fallback failed for {symbol_upper}: {exc}")

    yahoo = _yahoo_series(yahoo_points)
    yahoo_earliest = str(yahoo.index.min()) if not yahoo.empty else ""
    yahoo_latest = str(yahoo.index.max()) if not yahoo.empty else ""

    # Prefer IBKR on overlaps.
    combined = yahoo.copy()
    if not ibkr.empty:
        combined = pd.concat([combined, ibkr]).groupby(level=0).last()
    if combined.empty:
        coverage = {
            "symbol": symbol_upper,
            "required_start_date": minimum_required_start.isoformat(),
            "ibkr_earliest_date": ibkr_earliest,
            "yahoo_earliest_date": yahoo_earliest,
            "final_earliest_date": "",
            "final_latest_date": "",
            "row_count": 0,
            "missing_days_count": 0,
            "source_mix": "none",
            "status": "missing",
            "yahoo_provider_used": yahoo_provider_used,
            "yfinance_adapter_failure": yfinance_adapter_failure,
            "external_provider_available": external_provider_available,
        }
        return StitchedHistoryResult(values=[], coverage=coverage, warnings=warnings)

    combined = combined.sort_index()
    combined = combined[(combined.index >= preferred_start) & (combined.index <= as_of)]

    expected_days = pd.bdate_range(start=combined.index.min(), end=combined.index.max()).date
    expected = pd.Series(index=expected_days, dtype=float)
    expected.update(combined)
    expected = expected.dropna()

    row_count = int(len(expected))
    final_earliest = str(expected.index.min()) if row_count > 0 else ""
    final_latest = str(expected.index.max()) if row_count > 0 else ""
    missing_days_count = int(len(expected_days) - row_count) if row_count > 0 else 0

    ibkr_days = set(ibkr.index.tolist()) if not ibkr.empty else set()
    yahoo_days = set(yahoo.index.tolist()) if not yahoo.empty else set()
    merged_days = set(expected.index.tolist())
    ibkr_count = len(merged_days & ibkr_days)
    yahoo_count = len(merged_days & (yahoo_days - ibkr_days))
    source_mix = (
        "mixed" if ibkr_count > 0 and yahoo_count > 0
        else ("ibkr_only" if ibkr_count > 0 else ("yahoo_only" if yahoo_count > 0 else "none"))
    )

    status = "ok"
    if not final_earliest or _parse_date(final_earliest) > minimum_required_start:
        status = "insufficient"

    coverage = {
        "symbol": symbol_upper,
        "required_start_date": minimum_required_start.isoformat(),
        "ibkr_earliest_date": ibkr_earliest,
        "ibkr_latest_date": ibkr_latest,
        "yahoo_earliest_date": yahoo_earliest,
        "yahoo_latest_date": yahoo_latest,
        "final_earliest_date": final_earliest,
        "final_latest_date": final_latest,
        "row_count": row_count,
        "missing_days_count": missing_days_count,
        "source_mix": source_mix,
        "status": status,
        "yahoo_provider_used": yahoo_provider_used,
        "yfinance_adapter_failure": yfinance_adapter_failure,
        "external_provider_available": external_provider_available,
    }
    values = [float(item) for item in expected.tolist()]
    return StitchedHistoryResult(values=values, coverage=coverage, warnings=warnings)
