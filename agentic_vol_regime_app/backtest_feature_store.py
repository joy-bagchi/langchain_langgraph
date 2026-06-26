"""Backtest feature-store builder for deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import importlib.util
import json
import sqlite3
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import (
    DEFAULT_SECTOR_ETF_SYMBOLS,
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKRVolRegimeSnapshotRequest,
)
from agentic_vol_regime_app.features.sector_geometry import compute_sector_geometry_metrics
from src.data.historical_source_router import stitch_history_with_yahoo_fallback
from src.data.yahoo_eod_provider import YahooEODPoint, fetch_yahoo_eod_history_with_metadata


CORE_REGIME_SYMBOLS = ("SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M")
REQUIRED_BACKTEST_HISTORY_KEYS = (
    "SPY_close",
    "VIX",
    "VVIX",
    "VIX3M",
    *(f"{symbol}_close" for symbol in DEFAULT_SECTOR_ETF_SYMBOLS),
)
OPTIONAL_BACKTEST_HISTORY_KEYS = ("VIX9D", "VIX6M")
STRICT_10Y_TRAIN_LOOKBACK_DAYS = 2520
STRICT_10Y_COVERAGE_START = date(2013, 1, 1)
XLRE_INCEPTION_DATE = date(2015, 10, 8)
YAHOO_PREFERRED_START = "2010-01-01"
YAHOO_REQUIRED_EOD_MAP = {
    "SPY_close": "SPY",
    "VIX": "^VIX",
    "VVIX": "^VVIX",
    "VIX3M": "^VIX3M",
    **{f"{symbol}_close": symbol for symbol in DEFAULT_SECTOR_ETF_SYMBOLS},
}
YAHOO_OPTIONAL_EOD_MAP = {
    "VIX9D": "^VIX9D",
    "VIX6M": "^VIX6M",
    "XLC_close": "XLC",
}
HISTORICAL_DB_FILE_NAME = "historical_data.db"
STRICT_REPLAY_PRIMARY_YEARS = 10.0
STRICT_REPLAY_WARMUP_YEARS = 3.0
STRICT_REPLAY_MIN_PRIMARY_COVERAGE_RATIO = 0.75
STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS = int(
    252.0 * (STRICT_REPLAY_WARMUP_YEARS + (STRICT_REPLAY_PRIMARY_YEARS * STRICT_REPLAY_MIN_PRIMARY_COVERAGE_RATIO))
)
SECTOR_HISTORY_KEYS = tuple(f"{symbol}_close" for symbol in DEFAULT_SECTOR_ETF_SYMBOLS)


@dataclass(slots=True, frozen=True)
class FeatureStoreBuildResult:
    feature_store_path: str
    rows: int
    start_date: str
    end_date: str
    source_as_of: str
    warnings: list[str]
    history_coverage: list[dict[str, Any]] = field(default_factory=list)
    required_history_summary: dict[str, Any] = field(default_factory=dict)
    source_quality: dict[str, Any] = field(default_factory=dict)
    coverage_report_path: str | None = None


def _historical_db_path(*, app_paths: AppPaths) -> Path:
    return app_paths.root / "data" / "processed" / HISTORICAL_DB_FILE_NAME


def _ensure_historical_db_schema(*, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS eod_history (
                symbol TEXT NOT NULL,
                day TEXT NOT NULL,
                close REAL NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (symbol, day)
            )
            """
        )


def _load_cached_history_points(
    *,
    db_path: Path,
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[YahooEODPoint]:
    _ensure_historical_db_schema(db_path=db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT day, close, source
            FROM eod_history
            WHERE symbol = ? AND day >= ? AND day <= ?
            ORDER BY day ASC
            """,
            (str(symbol).upper(), start_date.isoformat(), end_date.isoformat()),
        ).fetchall()
    points: list[YahooEODPoint] = []
    for day_text, close_value, source in rows:
        parsed_day = pd.to_datetime(str(day_text)).date()
        points.append(YahooEODPoint(day=parsed_day, close=float(close_value), source=str(source)))
    return points


def _upsert_history_points(
    *,
    db_path: Path,
    symbol: str,
    points: list[YahooEODPoint],
) -> int:
    if not points:
        return 0
    _ensure_historical_db_schema(db_path=db_path)
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = [
        (
            str(symbol).upper(),
            point.day.isoformat(),
            float(point.close),
            str(point.source),
            now_iso,
        )
        for point in points
    ]
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO eod_history (symbol, day, close, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, day) DO UPDATE SET
                close = excluded.close,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            payload,
        )
    return len(payload)


def _compute_boundary_fetch_windows(
    *,
    cached_days: list[date],
    requested_start: date,
    requested_end: date,
) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    if requested_end < requested_start:
        return windows
    if not cached_days:
        return [(requested_start, requested_end)]
    earliest = min(cached_days)
    latest = max(cached_days)
    if earliest > requested_start:
        windows.append((requested_start, earliest - timedelta(days=1)))
    if latest < requested_end:
        windows.append((latest + timedelta(days=1), requested_end))
    normalized: list[tuple[date, date]] = []
    for start, end in windows:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        if end_date >= start_date:
            normalized.append((start_date, end_date))
    return normalized


def _build_series_from_points(
    *,
    points: list[YahooEODPoint],
    preferred_start: date,
    as_of: date,
) -> pd.Series:
    if not points:
        return pd.Series(dtype=float)
    by_date: dict[date, float] = {}
    for point in points:
        if preferred_start <= point.day <= as_of:
            by_date[point.day] = float(point.close)
    if not by_date:
        return pd.Series(dtype=float)
    ordered_days = sorted(by_date.keys())
    return pd.Series([float(by_date[day]) for day in ordered_days], index=ordered_days, dtype=float)


def _load_or_fetch_yahoo_history_with_cache(
    *,
    db_path: Path,
    symbol: str,
    preferred_start: date,
    as_of: date,
) -> tuple[list[YahooEODPoint], dict[str, Any], list[str]]:
    cached_points = _load_cached_history_points(
        db_path=db_path,
        symbol=symbol,
        start_date=preferred_start,
        end_date=as_of,
    )
    cached_days = [point.day for point in cached_points]
    windows = _compute_boundary_fetch_windows(
        cached_days=cached_days,
        requested_start=preferred_start,
        requested_end=as_of,
    )
    warnings: list[str] = []
    provider_used = "cache_only"
    yfinance_adapter_failure = False
    external_provider_available = bool(cached_points)
    inserted_rows = 0
    for start_date, end_date in windows:
        result = fetch_yahoo_eod_history_with_metadata(
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=(end_date + timedelta(days=1)).isoformat(),
        )
        warnings.extend(list(result.warnings))
        provider_used = str(result.provider_used)
        yfinance_adapter_failure = yfinance_adapter_failure or bool(result.yfinance_adapter_failure)
        external_provider_available = external_provider_available or bool(result.external_provider_available)
        inserted_rows += _upsert_history_points(
            db_path=db_path,
            symbol=symbol,
            points=list(result.points),
        )
    final_points = _load_cached_history_points(
        db_path=db_path,
        symbol=symbol,
        start_date=preferred_start,
        end_date=as_of,
    )
    metadata = {
        "provider_used": provider_used,
        "yfinance_adapter_failure": yfinance_adapter_failure,
        "external_provider_available": external_provider_available,
        "fetch_windows": [(start.isoformat(), end.isoformat()) for start, end in windows],
        "cache_rows_before": len(cached_points),
        "cache_rows_after": len(final_points),
        "inserted_rows": int(inserted_rows),
    }
    return final_points, metadata, warnings


def _safe_series(values: list[float], size: int) -> pd.Series:
    normalized = [float(item) for item in list(values) if item is not None]
    if len(normalized) >= size:
        return pd.Series(normalized[-size:], dtype=float)
    pad = [float("nan")] * max(size - len(normalized), 0)
    return pd.Series(pad + normalized, dtype=float)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    z = (series - mean) / std.replace({0.0: np.nan})
    return z.replace([np.inf, -np.inf], np.nan)


def _annualized_realized_vol(close_series: pd.Series, window: int) -> pd.Series:
    returns = np.log(close_series / close_series.shift(1))
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(252.0) * 100.0


def _rolling_drawdown(close_series: pd.Series, window: int) -> pd.Series:
    rolling_peak = close_series.rolling(window=window, min_periods=window).max()
    drawdown = (rolling_peak - close_series) / rolling_peak.replace({0.0: np.nan})
    return drawdown.clip(lower=0.0)


def _rolling_trend_persistence(close_series: pd.Series, window: int) -> pd.Series:
    up_steps = close_series.diff().ge(0.0).astype(float)
    return up_steps.rolling(window=window, min_periods=window).mean()


def _estimate_history_start_date(*, as_of: date, row_count: int) -> str:
    if int(row_count) <= 0:
        return ""
    return str(pd.bdate_range(end=as_of, periods=int(row_count)).date.min())


def build_feature_store_frame_from_observation(observation: ObservationRecord) -> pd.DataFrame:
    history = {str(key): list(value) for key, value in dict(observation.history).items()}
    required_history_keys = REQUIRED_BACKTEST_HISTORY_KEYS
    missing = [key for key in required_history_keys if not history.get(key)]
    if missing:
        raise RuntimeError(
            "Cannot build backtest feature store because observation history is missing keys: "
            + ", ".join(missing)
        )

    min_len = min(len(history[key]) for key in required_history_keys)
    if min_len < 64:
        raise RuntimeError(
            f"Backtest feature store requires at least 64 history rows; only received {min_len}."
        )

    as_of = pd.to_datetime(observation.as_of).date()
    dates = pd.bdate_range(end=as_of, periods=min_len).date
    frame = pd.DataFrame({"date": dates})
    frame["spy_close"] = _safe_series(history["SPY_close"], min_len).to_numpy()
    frame["vix"] = _safe_series(history["VIX"], min_len).to_numpy()
    frame["vvix"] = _safe_series(history["VVIX"], min_len).to_numpy()
    frame["vix9d"] = _safe_series(history.get("VIX9D", []), min_len).to_numpy()
    frame["vix3m"] = _safe_series(history["VIX3M"], min_len).to_numpy()
    frame["vix6m"] = _safe_series(history.get("VIX6M", []), min_len).to_numpy()
    frame["vix9m"] = _safe_series(history.get("VIX9M", []), min_len).to_numpy()

    for symbol in DEFAULT_SECTOR_ETF_SYMBOLS:
        key = f"{symbol}_close"
        values = history.get(key)
        if values and len(values) >= min_len:
            frame[key] = _safe_series(values, min_len).to_numpy()
        else:
            frame[key] = np.nan

    frame["spy_return_1d"] = frame["spy_close"].pct_change()
    frame["realized_vol_5d"] = _annualized_realized_vol(frame["spy_close"], window=5)
    frame["realized_vol_21d"] = _annualized_realized_vol(frame["spy_close"], window=21)
    frame["vvix_vix_ratio"] = frame["vvix"] / frame["vix"].replace({0.0: np.nan})
    frame["vix_z_22d"] = _rolling_zscore(frame["vix"], window=22)
    frame["vvix_vix_z_22d"] = _rolling_zscore(frame["vvix_vix_ratio"], window=22)
    frame["vix9d_vix_ratio"] = frame["vix9d"] / frame["vix"].replace({0.0: np.nan})
    frame["vix_vix3m_ratio"] = frame["vix"] / frame["vix3m"].replace({0.0: np.nan})
    frame["term_structure_slope"] = frame["vix3m"] - frame["vix"]
    frame["drawdown_21d"] = _rolling_drawdown(frame["spy_close"], window=21)
    frame["trend_persistence_21d"] = _rolling_trend_persistence(frame["spy_close"], window=21)

    avg_corr: list[float | None] = []
    first_share: list[float | None] = []
    eff_rank: list[float | None] = []
    log_det: list[float | None] = []
    sector_count_used: list[float | None] = []
    xlre_available: list[float | None] = []
    geometry_universe_version: list[float | None] = []
    for index in range(min_len):
        partial = {
            key: list(frame[key].iloc[: index + 1].astype(float).values)
            for key in frame.columns
            if key.endswith("_close")
        }
        metrics, _warnings = compute_sector_geometry_metrics(partial, lookback_days=21)
        avg_corr.append(metrics.get("avg_pairwise_corr_21d"))
        first_share.append(metrics.get("first_eigenvalue_share_21d"))
        eff_rank.append(metrics.get("effective_rank_21d"))
        log_det.append(metrics.get("log_det_corr_21d"))
        sector_count_used.append(metrics.get("sector_count_used"))
        xlre_available.append(metrics.get("xlre_available"))
        geometry_universe_version.append(metrics.get("geometry_universe_version"))
    frame["avg_pairwise_corr_21d"] = avg_corr
    frame["first_eigenvalue_share_21d"] = first_share
    frame["effective_rank_21d"] = eff_rank
    frame["log_det_corr_21d"] = log_det
    frame["sector_count_used"] = sector_count_used
    frame["xlre_available"] = xlre_available
    frame["geometry_universe_version"] = geometry_universe_version

    required_for_replay = [
        "spy_close",
        "spy_return_1d",
        "vix",
        "vvix",
        "realized_vol_5d",
        "realized_vol_21d",
        "vvix_vix_ratio",
        "vix_z_22d",
        "vvix_vix_z_22d",
        "vix_vix3m_ratio",
        "term_structure_slope",
        "drawdown_21d",
        "trend_persistence_21d",
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required_for_replay).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Feature-store build produced no usable rows after preprocessing.")
    return frame


def _persist_frame(frame: pd.DataFrame, output_path: Path) -> tuple[Path, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        frame.to_parquet(output_path, index=False)
        return output_path, warnings
    except Exception as exc:
        csv_path = output_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        warnings.append(
            "Failed to write parquet; wrote CSV instead. "
            f"Reason: {exc}"
        )
        return csv_path, warnings


def _persist_coverage_report(rows: list[dict[str, Any]], *, output_path: Path) -> Path:
    coverage_path = output_path.parent / "coverage_report.csv"
    pd.DataFrame(rows).to_csv(coverage_path, index=False)
    return coverage_path


def _is_xlre_late_inception_row(
    row: dict[str, Any],
    *,
    required_start_for_sectors: str,
) -> bool:
    symbol = str(row.get("symbol", "")).upper()
    if symbol != "XLRE":
        return False
    final_earliest_text = str(row.get("final_earliest_date", "")).strip()
    if not final_earliest_text:
        return False
    try:
        final_earliest = pd.to_datetime(final_earliest_text).date()
        required_start = pd.to_datetime(required_start_for_sectors).date()
    except Exception:
        return False
    return final_earliest > required_start and final_earliest >= XLRE_INCEPTION_DATE


def _is_sector_history_row(row: dict[str, Any]) -> bool:
    history_key = str(row.get("history_key", "")).strip()
    return history_key in SECTOR_HISTORY_KEYS


def _sector_row_meets_relaxed_coverage(row: dict[str, Any]) -> bool:
    row_count = int(row.get("row_count", 0) or 0)
    return row_count >= STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS


def _build_observation_from_history(
    *,
    as_of: date,
    history_payload: dict[str, list[float]],
    source: str,
    quality_warnings: list[str],
    provider_metadata: dict[str, Any],
) -> ObservationRecord:
    symbols_payload: dict[str, dict[str, Any]] = {}
    for key, values in history_payload.items():
        if not values:
            continue
        if key.endswith("_close"):
            symbol = key[:-6]
        else:
            symbol = key
        close_value = float(values[-1])
        symbols_payload[symbol] = {
            "last": close_value,
            "close": close_value,
            "bid": None,
            "ask": None,
            "volume": None,
        }
    as_of_iso = datetime.combine(as_of, time(21, 0, 0), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return ObservationRecord(
        schema_version="observation.v1",
        as_of=as_of_iso,
        source=source,
        symbols=symbols_payload,
        history=history_payload,
        quality={
            "is_complete": True,
            "warnings": list(quality_warnings),
            "missing_symbols": [],
            "missing_history": [],
            "stale_fields": [],
        },
        option_chain={},
        provider_metadata=dict(provider_metadata),
    )


def build_backtest_feature_store_from_ibkr(
    *,
    app_paths: AppPaths | None = None,
    output_path: str | Path | None = None,
    symbol: str = "SPY",
    history_days: int = 1512,
    as_of_date: str | None = None,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    market_data_type: int = 1,
    exchange: str = "SMART",
    option_exchange: str = "SMART",
    index_exchange: str = "CBOE",
    currency: str = "USD",
    minimum_required_sector_start_date: str | None = None,
    preferred_sector_start_date: str = "2010-01-01",
    ibkr_data_pipe: IBKRDataPipe | None = None,
) -> FeatureStoreBuildResult:
    paths = app_paths or AppPaths.default()
    resolved_output = Path(output_path) if output_path else paths.root / "data" / "processed" / "features_daily.parquet"
    pipe = ibkr_data_pipe or IBKRDataPipe(
        connection=IBKRConnectionConfig(
            host=host,
            port=int(port),
            client_id=int(client_id),
            readonly=True,
            timeout_seconds=10.0,
            market_data_type=int(market_data_type),
        )
    )
    long_seed_mode = int(history_days) >= STRICT_10Y_TRAIN_LOOKBACK_DAYS or bool(minimum_required_sector_start_date)
    coverage_rows: list[dict[str, Any]] = []
    fallback_warnings: list[str] = []
    source_warnings: list[str] = []
    source_quality: dict[str, Any] = {}
    history_payload: dict[str, list[float]] = {}
    if as_of_date:
        as_of = pd.to_datetime(str(as_of_date).strip()).date()
    else:
        as_of = datetime.now(timezone.utc).date()

    if long_seed_mode:
        if importlib.util.find_spec("yfinance") is None:
            raise RuntimeError(
                "Long-history EOD seed mode requires the optional 'yfinance' package, "
                "but it is not installed in this environment. "
                "Install it with: pip install yfinance"
            )
        historical_db_path = _historical_db_path(app_paths=paths)
        _ensure_historical_db_schema(db_path=historical_db_path)
        required_start_for_sectors = str(minimum_required_sector_start_date or STRICT_10Y_COVERAGE_START.isoformat())
        # Long-history seed: Yahoo first for EOD history.
        for history_key, yahoo_symbol in YAHOO_REQUIRED_EOD_MAP.items():
            preferred_start = pd.to_datetime(YAHOO_PREFERRED_START).date()
            cached_points, cache_metadata, cache_warnings = _load_or_fetch_yahoo_history_with_cache(
                db_path=historical_db_path,
                symbol=yahoo_symbol,
                preferred_start=preferred_start,
                as_of=as_of,
            )
            series = _build_series_from_points(points=cached_points, preferred_start=preferred_start, as_of=as_of)
            history_payload[history_key] = [float(item) for item in series.tolist()]
            final_earliest = str(series.index.min()) if not series.empty else ""
            final_latest = str(series.index.max()) if not series.empty else ""
            minimum_required_start = pd.to_datetime(required_start_for_sectors).date()
            coverage_row = {
                "symbol": str(yahoo_symbol).upper(),
                "required_start_date": minimum_required_start.isoformat(),
                "ibkr_earliest_date": "",
                "ibkr_latest_date": "",
                "yahoo_earliest_date": final_earliest,
                "yahoo_latest_date": final_latest,
                "final_earliest_date": final_earliest,
                "final_latest_date": final_latest,
                "row_count": int(len(series)),
                "missing_days_count": 0,
                "source_mix": "yahoo_only" if not series.empty else "none",
                "status": (
                    "ok"
                    if final_earliest and pd.to_datetime(final_earliest).date() <= minimum_required_start
                    else ("missing" if series.empty else "insufficient")
                ),
                "yahoo_provider_used": str(cache_metadata.get("provider_used", "cache_only")),
                "yfinance_adapter_failure": bool(cache_metadata.get("yfinance_adapter_failure", False)),
                "external_provider_available": bool(cache_metadata.get("external_provider_available", False)),
                "cache_rows_before": int(cache_metadata.get("cache_rows_before", 0)),
                "cache_rows_after": int(cache_metadata.get("cache_rows_after", 0)),
                "cache_inserted_rows": int(cache_metadata.get("inserted_rows", 0)),
                "cache_fetch_windows": json.dumps(list(cache_metadata.get("fetch_windows", []))),
            }
            coverage_row["history_key"] = history_key
            coverage_rows.append(coverage_row)
            fallback_warnings.extend(list(cache_warnings))
        for history_key, yahoo_symbol in YAHOO_OPTIONAL_EOD_MAP.items():
            preferred_start = pd.to_datetime(YAHOO_PREFERRED_START).date()
            cached_points, cache_metadata, cache_warnings = _load_or_fetch_yahoo_history_with_cache(
                db_path=historical_db_path,
                symbol=yahoo_symbol,
                preferred_start=preferred_start,
                as_of=as_of,
            )
            series = _build_series_from_points(points=cached_points, preferred_start=preferred_start, as_of=as_of)
            if not series.empty:
                history_payload[history_key] = [float(item) for item in series.tolist()]
            final_earliest = str(series.index.min()) if not series.empty else ""
            final_latest = str(series.index.max()) if not series.empty else ""
            minimum_required_start = pd.to_datetime(YAHOO_PREFERRED_START).date()
            coverage_row = {
                "symbol": str(yahoo_symbol).upper(),
                "required_start_date": minimum_required_start.isoformat(),
                "ibkr_earliest_date": "",
                "ibkr_latest_date": "",
                "yahoo_earliest_date": final_earliest,
                "yahoo_latest_date": final_latest,
                "final_earliest_date": final_earliest,
                "final_latest_date": final_latest,
                "row_count": int(len(series)),
                "missing_days_count": 0,
                "source_mix": "yahoo_only" if not series.empty else "none",
                "status": (
                    "ok"
                    if final_earliest and pd.to_datetime(final_earliest).date() <= minimum_required_start
                    else ("missing" if series.empty else "insufficient")
                ),
                "yahoo_provider_used": str(cache_metadata.get("provider_used", "cache_only")),
                "yfinance_adapter_failure": bool(cache_metadata.get("yfinance_adapter_failure", False)),
                "external_provider_available": bool(cache_metadata.get("external_provider_available", False)),
                "cache_rows_before": int(cache_metadata.get("cache_rows_before", 0)),
                "cache_rows_after": int(cache_metadata.get("cache_rows_after", 0)),
                "cache_inserted_rows": int(cache_metadata.get("inserted_rows", 0)),
                "cache_fetch_windows": json.dumps(list(cache_metadata.get("fetch_windows", []))),
            }
            coverage_row["history_key"] = history_key
            coverage_row["is_optional"] = True
            coverage_rows.append(coverage_row)
            fallback_warnings.extend(list(cache_warnings))

        # IBKR is best-effort recent overlay only; never fail long seed if this times out.
        try:
            recent_request_payload: dict[str, Any] = {
                "symbol": "SPY",
                "exchange": exchange,
                "option_exchange": option_exchange,
                "index_exchange": index_exchange,
                "currency": currency,
                "history_days": 10,
                "regime_symbols": ["SPY", "VIX", "VVIX", "VIX3M", *DEFAULT_SECTOR_ETF_SYMBOLS],
                "expiry_count": 0,
                "strike_count": 0,
                "min_days_to_expiry": 0,
                "as_of_date": str(as_of),
            }
            recent_request = IBKRVolRegimeSnapshotRequest.from_payload(recent_request_payload)
            recent_observation = pipe.fetch_vol_regime_snapshot(recent_request)
            source_quality = dict(recent_observation.quality)
            for key, ibkr_values in dict(recent_observation.history).items():
                key_text = str(key)
                if key_text not in history_payload:
                    continue
                merged = list(history_payload.get(key_text, []))
                tail = [float(item) for item in list(ibkr_values) if item is not None]
                if not tail:
                    continue
                overlay_len = min(len(merged), len(tail))
                if overlay_len > 0:
                    merged[-overlay_len:] = tail[-overlay_len:]
                    history_payload[key_text] = merged
            _time.sleep(12.0)
        except Exception as exc:
            source_warnings.append(f"IBKR recent overlay unavailable; using Yahoo EOD seed only. Reason: {exc}")
            source_quality = {
                "is_complete": False,
                "warnings": list(source_warnings),
                "missing_symbols": [],
                "missing_history": [],
                "stale_fields": [],
            }
        observation = _build_observation_from_history(
            as_of=as_of,
            history_payload=history_payload,
            source="YAHOO_EOD_PRIMARY_WITH_IBKR_RECENT_OVERLAY",
            quality_warnings=source_warnings,
            provider_metadata={
                "history_source_mode": "yahoo_primary_ibkr_recent_overlay",
                "requested_history_days": int(history_days),
                "as_of_date": str(as_of),
            },
        )
    else:
        request_payload: dict[str, Any] = {
            "symbol": symbol.upper().strip() or "SPY",
            "exchange": exchange,
            "option_exchange": option_exchange,
            "index_exchange": index_exchange,
            "currency": currency,
            "history_days": int(history_days),
            "regime_symbols": [symbol.upper().strip() or "SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", *DEFAULT_SECTOR_ETF_SYMBOLS],
            "expiry_count": 0,
            "strike_count": 0,
            "min_days_to_expiry": 0,
        }
        if as_of_date:
            request_payload["as_of_date"] = str(as_of_date).strip()

        request = IBKRVolRegimeSnapshotRequest.from_payload(request_payload)
        observation = pipe.fetch_vol_regime_snapshot(request)
        source_quality = dict(observation.quality)
        source_warnings = [str(item) for item in list(source_quality.get("warnings", []))]
        if list(source_quality.get("missing_symbols", [])):
            source_warnings.append(
                "IBKR missing symbols: " + ", ".join(str(item) for item in list(source_quality.get("missing_symbols", [])))
            )
        if list(source_quality.get("missing_history", [])):
            source_warnings.append(
                "IBKR missing history: " + ", ".join(str(item) for item in list(source_quality.get("missing_history", [])))
            )
        history_payload = {str(key): list(value) for key, value in dict(observation.history).items()}
    required_start_for_sectors = str(minimum_required_sector_start_date or "2013-01-01")
    if not long_seed_mode:
        for symbol_code in DEFAULT_SECTOR_ETF_SYMBOLS:
            key = f"{symbol_code}_close"
            stitched = stitch_history_with_yahoo_fallback(
                symbol=symbol_code,
                ibkr_values=list(history_payload.get(key, [])),
                as_of_date=as_of,
                preferred_start_date=str(preferred_sector_start_date),
                minimum_required_start_date=required_start_for_sectors,
            )
            history_payload[key] = list(stitched.values)
            coverage_row = dict(stitched.coverage)
            coverage_row["history_key"] = key
            coverage_rows.append(coverage_row)
            fallback_warnings.extend(list(stitched.warnings))

    resolved_output_abs = resolved_output.resolve()
    coverage_report_path = _persist_coverage_report(coverage_rows, output_path=resolved_output_abs)
    insufficient_coverage = [
        row
        for row in coverage_rows
        if str(row.get("status", "")) != "ok" and not bool(row.get("is_optional", False))
    ]
    xlre_soft_exemptions = [
        row
        for row in insufficient_coverage
        if _is_xlre_late_inception_row(row, required_start_for_sectors=required_start_for_sectors)
    ]
    if xlre_soft_exemptions:
        fallback_warnings.extend(
            [
                (
                    "XLRE starts after strict required start and is treated as late-inception; "
                    "pre-XLRE rows use 9-sector geometry and strict replay remains valid."
                )
                for _ in xlre_soft_exemptions
            ]
        )
        insufficient_coverage = [
            row
            for row in insufficient_coverage
            if row not in xlre_soft_exemptions
        ]
    sector_soft_exemptions: list[dict[str, Any]] = []
    sector_shortages: list[dict[str, Any]] = []
    for row in list(insufficient_coverage):
        if not _is_sector_history_row(row):
            continue
        if _sector_row_meets_relaxed_coverage(row):
            sector_soft_exemptions.append(row)
        else:
            sector_shortages.append(row)
    if sector_soft_exemptions:
        fallback_warnings.extend(
            [
                (
                    "Sector history relaxed coverage applied: "
                    f"symbol={str(row.get('symbol', 'unknown'))}, "
                    f"row_count={int(row.get('row_count', 0) or 0)}, "
                    f"minimum_required_rows={STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}."
                )
                for row in sector_soft_exemptions
            ]
        )
    if sector_shortages:
        fallback_warnings.extend(
            [
                (
                    "Sector history shortage retained as warning (non-fatal): "
                    f"symbol={str(row.get('symbol', 'unknown'))}, "
                    f"row_count={int(row.get('row_count', 0) or 0)}, "
                    f"minimum_required_rows={STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}, "
                    f"required_start_date={required_start_for_sectors}."
                )
                for row in sector_shortages
            ]
        )
    if sector_soft_exemptions or sector_shortages:
        removed = {id(row) for row in [*sector_soft_exemptions, *sector_shortages]}
        insufficient_coverage = [
            row
            for row in insufficient_coverage
            if id(row) not in removed
        ]
    if insufficient_coverage and minimum_required_sector_start_date:
        if all(int(row.get("row_count", 0) or 0) == 0 for row in insufficient_coverage):
            failed_symbols = ", ".join(str(item.get("symbol", "unknown")) for item in insufficient_coverage)
            provider_available = any(bool(row.get("external_provider_available", False)) for row in insufficient_coverage)
            adapter_failure = any(bool(row.get("yfinance_adapter_failure", False)) for row in insufficient_coverage)
            if provider_available:
                raise RuntimeError(
                    "Required long-history symbols still resolved to zero rows even though an external provider path was available. "
                    f"yfinance_adapter_failure={adapter_failure}. "
                    f"required_failed_symbols=[{failed_symbols}]. "
                    f"See coverage report: {coverage_report_path}"
                )
            raise RuntimeError(
                "External EOD history provider returned zero rows for all required long-history symbols. "
                "This is likely a network, firewall, proxy, or provider-access problem rather than a ticker-depth problem. "
                f"required_failed_symbols=[{failed_symbols}]. "
                f"See coverage report: {coverage_report_path}"
            )
        first = insufficient_coverage[0]
        symbol = str(first.get("symbol", "unknown"))
        final_earliest = str(first.get("final_earliest_date", ""))
        failed_symbols = ", ".join(str(item.get("symbol", "unknown")) for item in insufficient_coverage)
        raise RuntimeError(
            "Sector ETF history coverage is insufficient for strict replay. "
            f"symbol={symbol}, final_earliest_date={final_earliest}, "
            f"required_start_date={required_start_for_sectors}. "
            f"required_failed_symbols=[{failed_symbols}]. "
            f"See coverage report: {coverage_report_path}"
        )

    history_coverage: list[dict[str, Any]] = []
    for key in sorted(history_payload.keys()):
        row_count = len(history_payload.get(key, []))
        history_coverage.append(
            {
                "series": key,
                "rows": int(row_count),
                "inferred_start_date": _estimate_history_start_date(as_of=as_of, row_count=row_count),
                "inferred_end_date": str(as_of) if row_count > 0 else "",
                "is_required": key in REQUIRED_BACKTEST_HISTORY_KEYS,
            }
        )

    required_lengths = {key: len(history_payload.get(key, [])) for key in REQUIRED_BACKTEST_HISTORY_KEYS}
    min_required_rows = min(required_lengths.values()) if required_lengths else 0
    truncating_required_keys = sorted(
        [key for key, size in required_lengths.items() if int(size) == int(min_required_rows) and int(size) > 0]
    )
    required_history_summary = {
        "required_keys": list(REQUIRED_BACKTEST_HISTORY_KEYS),
        "optional_keys": list(OPTIONAL_BACKTEST_HISTORY_KEYS),
        "required_lengths": required_lengths,
        "min_required_rows": int(min_required_rows),
        "truncating_required_keys": truncating_required_keys,
        "inferred_required_start_date": _estimate_history_start_date(as_of=as_of, row_count=min_required_rows),
        "inferred_required_end_date": str(as_of) if min_required_rows > 0 else "",
    }

    patched_observation = ObservationRecord(
        schema_version=observation.schema_version,
        as_of=observation.as_of,
        source=observation.source,
        symbols=dict(observation.symbols),
        history=history_payload,
        quality=dict(observation.quality),
        option_chain=dict(observation.option_chain),
        provider_metadata=dict(observation.provider_metadata),
    )
    frame = build_feature_store_frame_from_observation(patched_observation)
    persisted_path, warnings = _persist_frame(frame, resolved_output_abs)
    combined_warnings = [*warnings, *source_warnings, *fallback_warnings]
    return FeatureStoreBuildResult(
        feature_store_path=str(persisted_path),
        rows=int(len(frame)),
        start_date=str(frame["date"].min()),
        end_date=str(frame["date"].max()),
        source_as_of=str(observation.as_of),
        warnings=combined_warnings,
        history_coverage=history_coverage,
        required_history_summary=required_history_summary,
        source_quality=source_quality,
        coverage_report_path=str(coverage_report_path),
    )
