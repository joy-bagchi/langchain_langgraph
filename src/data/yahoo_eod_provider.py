from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import quote

import pandas as pd


YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


@dataclass(slots=True, frozen=True)
class YahooEODPoint:
    day: date
    close: float
    source: str = "yahoo"


@dataclass(slots=True, frozen=True)
class YahooEODFetchResult:
    points: list[YahooEODPoint]
    provider_used: str
    yfinance_adapter_failure: bool
    external_provider_available: bool
    warnings: list[str]


def _to_unix_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(str(value)[:10]).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _frame_to_points(frame: pd.DataFrame, *, source: str) -> list[YahooEODPoint]:
    if frame.empty:
        return []
    preferred = "adj_close" if "adj_close" in frame.columns and frame["adj_close"].notna().any() else "close"
    points: list[YahooEODPoint] = []
    for row in frame.itertuples(index=False):
        day = pd.to_datetime(getattr(row, "date")).date()
        close_value = pd.to_numeric(pd.Series([getattr(row, preferred, None)]), errors="coerce").iloc[0]
        if pd.isna(close_value):
            continue
        close_float = float(close_value)
        if close_float <= 0.0:
            continue
        points.append(YahooEODPoint(day=day, close=close_float, source=source))
    return points


def fetch_raw_yahoo_chart_frame(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Raw Yahoo chart provider requires the optional 'requests' package.") from exc

    encoded_symbol = quote(str(symbol).strip().upper(), safe="")
    url = f"{YAHOO_CHART_BASE_URL}{encoded_symbol}"
    params = {
        "period1": _to_unix_seconds(start_date),
        "period2": _to_unix_seconds(end_date),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    chart = dict(payload.get("chart", {}))
    error_payload = chart.get("error")
    if error_payload:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error_payload}")
    results = list(chart.get("result", []) or [])
    if not results:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    result = dict(results[0])
    timestamps = list(result.get("timestamp", []) or [])
    indicators = dict(result.get("indicators", {}) or {})
    quote_rows = list(indicators.get("quote", []) or [])
    adjclose_rows = list(indicators.get("adjclose", []) or [])
    quote_block = dict(quote_rows[0]) if quote_rows else {}
    adjclose_block = dict(adjclose_rows[0]) if adjclose_rows else {}

    rows: list[dict[str, Any]] = []
    for index, ts in enumerate(timestamps):
        try:
            day = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        except Exception:
            continue
        rows.append(
            {
                "date": day.isoformat(),
                "open": pd.to_numeric(pd.Series([quote_block.get("open", [None])[index] if index < len(quote_block.get("open", []) or []) else None]), errors="coerce").iloc[0],
                "high": pd.to_numeric(pd.Series([quote_block.get("high", [None])[index] if index < len(quote_block.get("high", []) or []) else None]), errors="coerce").iloc[0],
                "low": pd.to_numeric(pd.Series([quote_block.get("low", [None])[index] if index < len(quote_block.get("low", []) or []) else None]), errors="coerce").iloc[0],
                "close": pd.to_numeric(pd.Series([quote_block.get("close", [None])[index] if index < len(quote_block.get("close", []) or []) else None]), errors="coerce").iloc[0],
                "adj_close": pd.to_numeric(pd.Series([adjclose_block.get("adjclose", [None])[index] if index < len(adjclose_block.get("adjclose", []) or []) else None]), errors="coerce").iloc[0],
                "volume": pd.to_numeric(pd.Series([quote_block.get("volume", [None])[index] if index < len(quote_block.get("volume", []) or []) else None]), errors="coerce").iloc[0],
                "source": "yahoo_chart",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.dropna(subset=["close"], how="all").reset_index(drop=True)
    return frame


def _fetch_yfinance_frame(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Yahoo fallback requires the optional 'yfinance' package.") from exc

    try:
        cache_root = Path(tempfile.gettempdir()) / "yfinance_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_root))
    except Exception:
        pass

    symbol_text = str(symbol).strip().upper()
    frame = yf.download(
        tickers=symbol_text,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    working = pd.DataFrame(frame).copy() if frame is not None else pd.DataFrame()
    # yfinance may return MultiIndex columns even for one ticker.
    # Keep only one level of OHLCV names so downstream coercion stays 1-D.
    if isinstance(working.columns, pd.MultiIndex) and len(working.columns) > 0:
        level0 = [str(item) for item in list(working.columns.get_level_values(0))]
        level1 = [str(item) for item in list(working.columns.get_level_values(1))]
        ohlcv_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if set(level0) & ohlcv_names:
            working.columns = level0
        elif set(level1) & ohlcv_names:
            working.columns = level1
        else:
            working.columns = [str(item[0]) for item in list(working.columns)]
    if working.empty:
        ticker = yf.Ticker(symbol_text)
        fallback = ticker.history(
            start=str(start_date),
            end=str(end_date),
            interval="1d",
            auto_adjust=False,
            actions=False,
        )
        working = pd.DataFrame(fallback).copy() if fallback is not None else pd.DataFrame()
    if working.empty and symbol_text in {"^VIX", "^VVIX", "^VIX3M"}:
        aliases = {"^VIX": ["VIX"], "^VVIX": ["VVIX"], "^VIX3M": ["VIX3M"]}
        for alias in aliases.get(symbol_text, []):
            try:
                ticker = yf.Ticker(alias)
                fallback = ticker.history(
                    start=str(start_date),
                    end=str(end_date),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )
            except Exception:
                continue
            candidate = pd.DataFrame(fallback).copy() if fallback is not None else pd.DataFrame()
            if not candidate.empty:
                working = candidate
                break
    if working.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "source"])
    working = working.reset_index()
    date_column = "Date" if "Date" in working.columns else working.columns[0]
    def _col_or_na(column_name: str) -> pd.Series:
        if column_name not in working.columns:
            return pd.Series([pd.NA] * len(working), dtype="object")
        column = working[column_name]
        if isinstance(column, pd.DataFrame):
            # Defensive fallback: take first column if residual duplicate columns remain.
            column = column.iloc[:, 0]
        return pd.to_numeric(column, errors="coerce")

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(working[date_column]).dt.date,
            "open": _col_or_na("Open"),
            "high": _col_or_na("High"),
            "low": _col_or_na("Low"),
            "close": _col_or_na("Close"),
            "adj_close": _col_or_na("Adj Close"),
            "volume": _col_or_na("Volume"),
            "source": "yfinance",
        }
    )
    normalized = normalized.dropna(subset=["close"], how="all").reset_index(drop=True)
    return normalized


def fetch_yahoo_eod_history(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[YahooEODPoint]:
    return fetch_yahoo_eod_history_with_metadata(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    ).points


def fetch_yahoo_eod_history_with_metadata(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> YahooEODFetchResult:
    warnings: list[str] = []
    yfinance_frame = pd.DataFrame()
    try:
        yfinance_frame = _fetch_yfinance_frame(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        warnings.append(f"yfinance adapter failed for {symbol}: {exc}")
        yfinance_frame = pd.DataFrame()
    if not yfinance_frame.empty:
        return YahooEODFetchResult(
            points=_frame_to_points(yfinance_frame, source="yfinance"),
            provider_used="yfinance",
            yfinance_adapter_failure=False,
            external_provider_available=True,
            warnings=warnings,
        )

    raw_frame = pd.DataFrame()
    try:
        raw_frame = fetch_raw_yahoo_chart_frame(symbol=symbol, start_date=start_date, end_date=end_date)
    except Exception as exc:
        warnings.append(f"raw yahoo chart provider failed for {symbol}: {exc}")
        raw_frame = pd.DataFrame()
    if not raw_frame.empty:
        return YahooEODFetchResult(
            points=_frame_to_points(raw_frame, source="yahoo_chart"),
            provider_used="yahoo_chart",
            yfinance_adapter_failure=True,
            external_provider_available=True,
            warnings=warnings,
        )

    return YahooEODFetchResult(
        points=[],
        provider_used="none",
        yfinance_adapter_failure=not yfinance_frame.empty or bool(warnings),
        external_provider_available=False,
        warnings=warnings,
    )
