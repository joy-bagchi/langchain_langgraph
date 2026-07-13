from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.data.ibkr_client import (
    DEFAULT_SECTOR_ETF_SYMBOLS,
    IBKRConnectionConfig,
    IBKRDailyBar,
    IBKRDailyHistoryRequest,
    IBKRDataPipe,
)


SECTOR_PRICE_SCHEMA_VERSION = "sector_prices.v1"
DEFAULT_SECTOR_PRICE_SYMBOLS = (*DEFAULT_SECTOR_ETF_SYMBOLS, "SPY")
DEFAULT_OVERLAP_TRADING_DAYS = 5
DEFAULT_REPAIR_HORIZON_TRADING_DAYS = 10
DEFAULT_MARKET_CLOSE_CUTOFF = time(hour=16, minute=15)
DEFAULT_MARKET_TIMEZONE = "America/New_York"


def default_sector_history_paths(
    app_paths: AppPaths | None = None,
) -> tuple[Path, Path]:
    paths = app_paths or AppPaths.default()
    base_dir = paths.root / "data" / "market_history"
    return (
        base_dir / "sector_prices_daily.parquet",
        base_dir / "sector_prices_daily.metadata.json",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _normalize_symbol_universe(symbols: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    configured = symbols or DEFAULT_SECTOR_PRICE_SYMBOLS
    normalized = []
    seen: set[str] = set()
    for item in configured:
        symbol = str(item).strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return tuple(normalized)


def _normalize_explicit_target_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        normalized = value
    else:
        normalized = date.fromisoformat(str(value).strip()[:10])
    while normalized.weekday() >= 5:
        normalized -= timedelta(days=1)
    return normalized


def subtract_weekday_sessions(day: date, sessions: int) -> date:
    cursor = day
    remaining = max(int(sessions), 0)
    while remaining > 0:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return cursor


def business_day_start_from_period(end_date: date, trading_days: int) -> date:
    periods = max(int(trading_days), 1)
    return pd.bdate_range(end=end_date, periods=periods).date.min()


def resolve_target_completed_session(
    *,
    explicit_target_end_date: date | str | None = None,
    now: datetime | None = None,
    market_timezone: str = DEFAULT_MARKET_TIMEZONE,
    market_close_cutoff: time = DEFAULT_MARKET_CLOSE_CUTOFF,
) -> date:
    explicit = _normalize_explicit_target_date(explicit_target_end_date)
    if explicit is not None:
        return explicit
    current = now or datetime.now(timezone.utc)
    localized = current.astimezone(ZoneInfo(market_timezone))
    session_day = localized.date()
    if session_day.weekday() >= 5:
        return _normalize_explicit_target_date(session_day)
    if localized.timetz().replace(tzinfo=None) < market_close_cutoff:
        return subtract_weekday_sessions(session_day, 1)
    return session_day


def _safe_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or number <= 0.0:
        return None
    return float(number)


def _frame_content_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    csv_bytes = normalized.to_csv(index=False, na_rep="").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _parquet_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class SectorPriceStoreValidation:
    frame: pd.DataFrame
    first_date: str
    last_date: str
    row_count: int
    per_symbol: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SymbolSyncResult:
    symbol: str
    status: str
    previous_last_date: str | None
    requested_start: str | None
    requested_end: str | None
    received: int
    inserted: int
    revised: int
    new_last_date: str | None
    actual_what_to_show: str | None = None
    request_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "previous_last_date": self.previous_last_date,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "received": int(self.received),
            "inserted": int(self.inserted),
            "revised": int(self.revised),
            "new_last_date": self.new_last_date,
            "actual_what_to_show": self.actual_what_to_show,
            "request_count": int(self.request_count),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class SectorHistorySyncResult:
    mode: str
    store_before: dict[str, Any]
    store_after: dict[str, Any]
    ibkr_request_count: int
    symbols: dict[str, SymbolSyncResult]
    warnings: list[str]
    parquet_path: str
    metadata_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "store_before": dict(self.store_before),
            "store_after": dict(self.store_after),
            "ibkr_request_count": int(self.ibkr_request_count),
            "symbols": {symbol: result.to_dict() for symbol, result in self.symbols.items()},
            "warnings": list(self.warnings),
            "parquet_path": self.parquet_path,
            "metadata_path": self.metadata_path,
        }


class SectorPriceStore:
    def __init__(
        self,
        *,
        parquet_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        default_parquet, default_metadata = default_sector_history_paths()
        self.parquet_path = Path(parquet_path or default_parquet).resolve()
        self.metadata_path = Path(metadata_path or default_metadata).resolve()
        self.symbols = _normalize_symbol_universe(symbols)

    def exists(self) -> bool:
        return self.parquet_path.exists() and self.metadata_path.exists()

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Sector history metadata does not exist: {self.metadata_path}")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def read_frame(self) -> pd.DataFrame:
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Sector history parquet does not exist: {self.parquet_path}")
        return pd.read_parquet(self.parquet_path)

    def load_offline(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(
                "Offline sector history requires an existing local store. "
                f"Missing files: parquet={self.parquet_path.exists()} metadata={self.metadata_path.exists()}."
            )
        frame = self.read_frame()
        validation = self.validate_frame(frame)
        metadata = self.load_metadata()
        expected_sha = str(metadata.get("content_sha256", "")).strip()
        if expected_sha and expected_sha != _frame_content_sha256(validation.frame):
            raise ValueError("Stored sector history metadata checksum does not match the parquet content.")
        return validation.frame

    def summarize_existing_store(self) -> dict[str, Any]:
        if not self.exists():
            return {"row_count": 0, "first_date": None, "last_date": None}
        validation = self.validate_frame(self.read_frame())
        return {
            "row_count": int(validation.row_count),
            "first_date": validation.first_date,
            "last_date": validation.last_date,
        }

    def validate_frame(self, frame: pd.DataFrame) -> SectorPriceStoreValidation:
        if "date" not in frame.columns:
            raise ValueError("Sector history dataset must contain a 'date' column.")
        expected_columns = ["date", *self.symbols]
        if list(frame.columns) != expected_columns:
            raise ValueError(
                "Sector history columns do not match the configured symbol universe. "
                f"expected={expected_columns} actual={list(frame.columns)}"
            )
        normalized = frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.normalize()
        if normalized["date"].duplicated().any():
            raise ValueError("Sector history dataset contains duplicate session dates.")
        normalized = normalized.sort_values("date", kind="stable").reset_index(drop=True)
        if not normalized["date"].is_monotonic_increasing:
            raise ValueError("Sector history dataset must be sorted by ascending date.")

        per_symbol: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        date_index = normalized["date"]
        for symbol in self.symbols:
            series = pd.to_numeric(normalized[symbol], errors="coerce")
            invalid_non_null = normalized[symbol].notna() & (~series.isna()) & (series <= 0.0)
            if invalid_non_null.any():
                raise ValueError(f"Sector history contains non-positive values for {symbol}.")
            infinite_mask = normalized[symbol].map(lambda value: isinstance(value, float) and math.isinf(value))
            if infinite_mask.any():
                raise ValueError(f"Sector history contains Infinity values for {symbol}.")
            normalized[symbol] = series.astype(float)
            non_null_mask = normalized[symbol].notna()
            first_valid = date_index[non_null_mask].min() if non_null_mask.any() else pd.NaT
            last_valid = date_index[non_null_mask].max() if non_null_mask.any() else pd.NaT
            internal_gap_count = 0
            if non_null_mask.any():
                coverage_mask = (date_index >= first_valid) & (date_index <= last_valid)
                internal_gap_count = int(normalized.loc[coverage_mask, symbol].isna().sum())
                if internal_gap_count:
                    warnings.append(f"{symbol} has {internal_gap_count} internal null gaps inside observed coverage.")
            per_symbol[symbol] = {
                "first_valid_date": first_valid.date().isoformat() if pd.notna(first_valid) else None,
                "last_valid_date": last_valid.date().isoformat() if pd.notna(last_valid) else None,
                "non_null_count": int(non_null_mask.sum()),
                "internal_gap_count": int(internal_gap_count),
            }

        first_date = normalized["date"].min()
        last_date = normalized["date"].max()
        normalized["date"] = pd.to_datetime(normalized["date"])
        return SectorPriceStoreValidation(
            frame=normalized,
            first_date=first_date.date().isoformat() if pd.notna(first_date) else "",
            last_date=last_date.date().isoformat() if pd.notna(last_date) else "",
            row_count=int(len(normalized)),
            per_symbol=per_symbol,
            warnings=warnings,
        )

    def build_metadata(
        self,
        *,
        frame: pd.DataFrame,
        mode: str,
        requested_settings: dict[str, Any],
        per_symbol_fetch: dict[str, SymbolSyncResult],
        warnings: list[str],
    ) -> dict[str, Any]:
        validation = self.validate_frame(frame)
        metadata_per_symbol: dict[str, dict[str, Any]] = {}
        for symbol in self.symbols:
            fetch_result = per_symbol_fetch.get(symbol)
            coverage = dict(validation.per_symbol.get(symbol, {}))
            metadata_per_symbol[symbol] = {
                **coverage,
                "actual_what_to_show": fetch_result.actual_what_to_show if fetch_result else None,
                "fetch_status": fetch_result.status if fetch_result else "offline",
                "requested_start": fetch_result.requested_start if fetch_result else None,
                "requested_end": fetch_result.requested_end if fetch_result else None,
                "received_bar_count": int(fetch_result.received if fetch_result else 0),
                "inserted_count": int(fetch_result.inserted if fetch_result else 0),
                "revised_count": int(fetch_result.revised if fetch_result else 0),
            }
        return {
            "schema_version": SECTOR_PRICE_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source": "IBKR",
            "mode": mode,
            "symbols": list(self.symbols),
            "requested_settings": requested_settings,
            "first_date": validation.first_date,
            "last_date": validation.last_date,
            "row_count": int(validation.row_count),
            "content_sha256": _frame_content_sha256(validation.frame),
            "per_symbol": metadata_per_symbol,
            "warnings": list(warnings),
        }

    def write_authoritative(
        self,
        *,
        frame: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        validation = self.validate_frame(frame)
        temp_parquet = self.parquet_path.with_suffix(f"{self.parquet_path.suffix}.tmp")
        temp_metadata = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")
        validation.frame.to_parquet(temp_parquet, index=False)
        reloaded = pd.read_parquet(temp_parquet)
        revalidated = self.validate_frame(reloaded)
        if revalidated.row_count != validation.row_count or list(revalidated.frame.columns) != list(validation.frame.columns):
            raise ValueError("Temporary parquet validation failed before publish.")
        payload = dict(metadata)
        payload["file_sha256"] = _parquet_file_sha256(temp_parquet)
        temp_metadata.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default), encoding="utf-8")
        os.replace(temp_parquet, self.parquet_path)
        os.replace(temp_metadata, self.metadata_path)


@dataclass(slots=True)
class _FetchedSymbolBars:
    bars: list[IBKRDailyBar]
    actual_what_to_show: str | None
    request_count: int
    warnings: list[str]


class IBKRHistoricalStoreUpdater:
    def __init__(
        self,
        *,
        store: SectorPriceStore,
        data_pipe: IBKRDataPipe | None = None,
        connection: IBKRConnectionConfig | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
        preferred_what_to_show: str = "ADJUSTED_LAST",
        allow_trades_fallback: bool = True,
        overlap_trading_days: int = DEFAULT_OVERLAP_TRADING_DAYS,
        repair_horizon_trading_days: int = DEFAULT_REPAIR_HORIZON_TRADING_DAYS,
        market_timezone: str = DEFAULT_MARKET_TIMEZONE,
        market_close_cutoff: time = DEFAULT_MARKET_CLOSE_CUTOFF,
        chunk_calendar_days: int = 365,
        use_rth: bool = True,
    ) -> None:
        self.store = store
        self.connection = connection or IBKRConnectionConfig()
        self.data_pipe = data_pipe or IBKRDataPipe(connection=self.connection)
        self.symbols = _normalize_symbol_universe(symbols or store.symbols)
        self.preferred_what_to_show = str(preferred_what_to_show).upper()
        self.allow_trades_fallback = bool(allow_trades_fallback)
        self.overlap_trading_days = max(int(overlap_trading_days), 0)
        self.repair_horizon_trading_days = max(int(repair_horizon_trading_days), 0)
        self.market_timezone = market_timezone
        self.market_close_cutoff = market_close_cutoff
        self.chunk_calendar_days = max(int(chunk_calendar_days), 30)
        self.use_rth = bool(use_rth)

    def bootstrap(
        self,
        *,
        start_date: date | str | None = None,
        history_years: int | None = None,
        history_days: int | None = None,
        target_end_date: date | str | None = None,
        force: bool = False,
    ) -> SectorHistorySyncResult:
        if self.store.exists() and not force:
            raise FileExistsError(
                f"Authoritative sector history store already exists at {self.store.parquet_path}. "
                "Use force=True to replace it."
            )
        target_end = resolve_target_completed_session(
            explicit_target_end_date=target_end_date,
            market_timezone=self.market_timezone,
            market_close_cutoff=self.market_close_cutoff,
        )
        bootstrap_start = self._resolve_bootstrap_start(
            target_end=target_end,
            start_date=start_date,
            history_years=history_years,
            history_days=history_days,
        )
        frame = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        symbol_results: dict[str, SymbolSyncResult] = {}
        warnings: list[str] = []
        ibkr_request_count = 0
        for symbol in self.symbols:
            fetched = self._fetch_symbol_range(symbol=symbol, request_start=bootstrap_start, request_end=target_end)
            ibkr_request_count += fetched.request_count
            warnings.extend(fetched.warnings)
            frame, inserted_count, revised_count = self._merge_symbol_bars(
                base_frame=frame,
                symbol=symbol,
                fetched_bars=fetched.bars,
            )
            symbol_results[symbol] = SymbolSyncResult(
                symbol=symbol,
                status="bootstrapped",
                previous_last_date=None,
                requested_start=bootstrap_start.isoformat(),
                requested_end=target_end.isoformat(),
                received=len(fetched.bars),
                inserted=inserted_count,
                revised=revised_count,
                new_last_date=self._last_valid_date_for_symbol_iso(frame, symbol),
                actual_what_to_show=fetched.actual_what_to_show,
                request_count=fetched.request_count,
                warnings=list(fetched.warnings),
            )
        frame = self._finalize_frame(frame)
        metadata = self.store.build_metadata(
            frame=frame,
            mode="bootstrap",
            requested_settings=self._requested_settings(),
            per_symbol_fetch=symbol_results,
            warnings=warnings,
        )
        self.store.write_authoritative(frame=frame, metadata=metadata)
        after = self.store.summarize_existing_store()
        return SectorHistorySyncResult(
            mode="bootstrap",
            store_before={"row_count": 0, "first_date": None, "last_date": None},
            store_after=after,
            ibkr_request_count=ibkr_request_count,
            symbols=symbol_results,
            warnings=warnings,
            parquet_path=str(self.store.parquet_path),
            metadata_path=str(self.store.metadata_path),
        )

    def update(
        self,
        *,
        target_end_date: date | str | None = None,
        repair_start_date: date | str | None = None,
        bootstrap_start_date: date | str | None = None,
    ) -> SectorHistorySyncResult:
        if not self.store.exists():
            raise FileNotFoundError("Update mode requires an existing authoritative sector history store.")
        existing = self.store.load_offline()
        before = self.store.summarize_existing_store()
        target_end = resolve_target_completed_session(
            explicit_target_end_date=target_end_date,
            market_timezone=self.market_timezone,
            market_close_cutoff=self.market_close_cutoff,
        )
        repair_start = _normalize_explicit_target_date(repair_start_date)
        full_start = _normalize_explicit_target_date(bootstrap_start_date)
        if full_start is None and not existing.empty:
            full_start = pd.to_datetime(existing["date"]).dt.date.min()
        if full_start is None:
            raise ValueError("Update mode requires bootstrap_start_date when the existing store has no rows.")

        frame = existing.copy()
        symbol_results: dict[str, SymbolSyncResult] = {}
        warnings: list[str] = []
        ibkr_request_count = 0
        for symbol in self.symbols:
            previous_last_date = self._last_valid_date_for_symbol(frame, symbol)
            previous_last = previous_last_date.isoformat() if previous_last_date is not None else None
            request_start = self._determine_update_start(
                frame=frame,
                symbol=symbol,
                target_end=target_end,
                repair_start_date=repair_start,
                bootstrap_start_date=full_start,
            )
            if request_start is None:
                symbol_results[symbol] = SymbolSyncResult(
                    symbol=symbol,
                    status="already_current",
                    previous_last_date=previous_last,
                    requested_start=None,
                    requested_end=None,
                    received=0,
                    inserted=0,
                    revised=0,
                    new_last_date=previous_last,
                    request_count=0,
                )
                continue
            fetched = self._fetch_symbol_range(symbol=symbol, request_start=request_start, request_end=target_end)
            ibkr_request_count += fetched.request_count
            warnings.extend(fetched.warnings)
            frame, inserted_count, revised_count = self._merge_symbol_bars(
                base_frame=frame,
                symbol=symbol,
                fetched_bars=[bar for bar in fetched.bars if request_start <= bar.session_date <= target_end],
            )
            symbol_results[symbol] = SymbolSyncResult(
                symbol=symbol,
                status="updated",
                previous_last_date=previous_last,
                requested_start=request_start.isoformat(),
                requested_end=target_end.isoformat(),
                received=len(fetched.bars),
                inserted=inserted_count,
                revised=revised_count,
                new_last_date=self._last_valid_date_for_symbol_iso(frame, symbol),
                actual_what_to_show=fetched.actual_what_to_show,
                request_count=fetched.request_count,
                warnings=list(fetched.warnings),
            )
        frame = self._finalize_frame(frame)
        metadata = self.store.build_metadata(
            frame=frame,
            mode="update",
            requested_settings=self._requested_settings(),
            per_symbol_fetch=symbol_results,
            warnings=warnings,
        )
        self.store.write_authoritative(frame=frame, metadata=metadata)
        after = self.store.summarize_existing_store()
        return SectorHistorySyncResult(
            mode="update",
            store_before=before,
            store_after=after,
            ibkr_request_count=ibkr_request_count,
            symbols=symbol_results,
            warnings=warnings,
            parquet_path=str(self.store.parquet_path),
            metadata_path=str(self.store.metadata_path),
        )

    def load_offline(self) -> pd.DataFrame:
        return self.store.load_offline()

    def validate(self) -> SectorPriceStoreValidation:
        return self.store.validate_frame(self.store.read_frame())

    def _requested_settings(self) -> dict[str, Any]:
        return {
            "bar_size": "1 day",
            "preferred_what_to_show": self.preferred_what_to_show,
            "use_rth": self.use_rth,
            "overlap_trading_days": self.overlap_trading_days,
            "repair_horizon_trading_days": self.repair_horizon_trading_days,
        }

    def _resolve_bootstrap_start(
        self,
        *,
        target_end: date,
        start_date: date | str | None,
        history_years: int | None,
        history_days: int | None,
    ) -> date:
        explicit_start = _normalize_explicit_target_date(start_date)
        if explicit_start is not None:
            return explicit_start
        if history_years is not None:
            return _normalize_explicit_target_date(target_end - timedelta(days=365 * max(int(history_years), 1)))
        if history_days is not None:
            return business_day_start_from_period(target_end, int(history_days))
        return date(2015, 1, 1)

    def _fetch_symbol_range(
        self,
        *,
        symbol: str,
        request_start: date,
        request_end: date,
    ) -> _FetchedSymbolBars:
        warnings: list[str] = []
        all_bars: dict[date, IBKRDailyBar] = {}
        request_count = 0
        actual_what_to_show_values: set[str] = set()
        cursor_end = request_end
        while cursor_end >= request_start:
            window_start = max(request_start, cursor_end - timedelta(days=self.chunk_calendar_days))
            duration_days = max((cursor_end - window_start).days + 10, 1)
            request = IBKRDailyHistoryRequest(
                symbol=symbol,
                end_date=cursor_end,
                duration_calendar_days=duration_days,
                exchange="SMART",
                currency="USD",
                use_rth=self.use_rth,
                preferred_what_to_show=self.preferred_what_to_show,
                allow_fallback_to_trades=self.allow_trades_fallback,
            )
            try:
                result = self.data_pipe.request_daily_bars(request)
            except RuntimeError as exc:
                message = str(exc)
                if all_bars and "no usable dated daily bars" in message.lower():
                    warnings.append(
                        f"{symbol} returned no earlier bars for {window_start.isoformat()} -> {cursor_end.isoformat()}; "
                        "treating this as the pre-inception boundary."
                    )
                    break
                raise
            request_count += 1
            actual_what_to_show_values.add(str(result.actual_what_to_show).upper())
            for warning in result.warnings:
                if warning not in warnings:
                    warnings.append(str(warning))
            if str(result.actual_what_to_show).upper() != self.preferred_what_to_show:
                warnings.append(
                    f"{symbol} fell back to {result.actual_what_to_show}; prices are not labeled adjusted."
                )
            for bar in result.bars:
                if window_start <= bar.session_date <= request_end:
                    all_bars[bar.session_date] = bar
            cursor_end = window_start - timedelta(days=1)
        actual_what_to_show = None
        if len(actual_what_to_show_values) == 1:
            actual_what_to_show = next(iter(actual_what_to_show_values))
        elif actual_what_to_show_values:
            actual_what_to_show = "MIXED"
            warnings.append(f"{symbol} used mixed whatToShow values across chunked requests: {sorted(actual_what_to_show_values)}.")
        return _FetchedSymbolBars(
            bars=[all_bars[item] for item in sorted(all_bars.keys())],
            actual_what_to_show=actual_what_to_show,
            request_count=request_count,
            warnings=warnings,
        )

    def _determine_update_start(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        target_end: date,
        repair_start_date: date | None,
        bootstrap_start_date: date,
    ) -> date | None:
        last_stored = self._last_valid_date_for_symbol(frame, symbol)
        recent_gap_start = self._recent_gap_start(
            frame=frame,
            symbol=symbol,
            repair_start_date=repair_start_date,
        )
        if last_stored is None:
            return recent_gap_start or bootstrap_start_date
        if last_stored >= target_end and recent_gap_start is None:
            return None
        overlap_start = subtract_weekday_sessions(last_stored, self.overlap_trading_days)
        starts = [overlap_start]
        if recent_gap_start is not None:
            starts.append(recent_gap_start)
        return min(starts)

    def _recent_gap_start(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        repair_start_date: date | None,
    ) -> date | None:
        if frame.empty or symbol not in frame.columns:
            return None
        normalized_dates = pd.to_datetime(frame["date"]).dt.date
        series = pd.to_numeric(frame[symbol], errors="coerce")
        non_null_mask = series.notna()
        if not non_null_mask.any():
            return None
        first_valid = normalized_dates[non_null_mask].min()
        last_valid = normalized_dates[non_null_mask].max()
        horizon_start = subtract_weekday_sessions(last_valid, self.repair_horizon_trading_days)
        effective_start = max(horizon_start, repair_start_date) if repair_start_date is not None else horizon_start
        coverage_mask = (normalized_dates >= first_valid) & (normalized_dates <= last_valid) & (normalized_dates >= effective_start)
        gap_dates = normalized_dates[coverage_mask & series.isna()]
        if gap_dates.empty:
            return None
        return min(gap_dates)

    def _merge_symbol_bars(
        self,
        *,
        base_frame: pd.DataFrame,
        symbol: str,
        fetched_bars: list[IBKRDailyBar],
    ) -> tuple[pd.DataFrame, int, int]:
        frame = base_frame.copy()
        if "date" not in frame.columns:
            frame = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        if symbol not in frame.columns:
            frame[symbol] = pd.Series(dtype=float)
        if frame.empty:
            frame = pd.DataFrame(columns=["date", *self.symbols])
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date", drop=True)
        inserted_count = 0
        revised_count = 0
        for bar in fetched_bars:
            value = _safe_positive_float(bar.close)
            if value is None:
                continue
            row_key = pd.Timestamp(bar.session_date)
            previous = frame.at[row_key, symbol] if row_key in frame.index else float("nan")
            if row_key not in frame.index:
                frame.loc[row_key, :] = pd.Series(dtype=float)
                previous = float("nan")
            if pd.isna(previous):
                inserted_count += 1
            elif float(previous) != float(value):
                revised_count += 1
            frame.at[row_key, symbol] = float(value)
        frame = frame.reset_index().rename(columns={"index": "date"})
        frame = self._finalize_frame(frame)
        return frame, inserted_count, revised_count

    def _finalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        if "date" not in normalized.columns:
            normalized["date"] = pd.Series(dtype="datetime64[ns]")
        normalized["date"] = pd.to_datetime(normalized["date"])
        for symbol in self.symbols:
            if symbol not in normalized.columns:
                normalized[symbol] = pd.Series(dtype=float)
            normalized[symbol] = pd.to_numeric(normalized[symbol], errors="coerce")
        normalized = normalized[["date", *self.symbols]].sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        return normalized

    @staticmethod
    def _last_valid_date_for_symbol(frame: pd.DataFrame, symbol: str) -> date | None:
        if frame.empty or symbol not in frame.columns:
            return None
        series = pd.to_numeric(frame[symbol], errors="coerce")
        if series.notna().sum() == 0:
            return None
        dates = pd.to_datetime(frame.loc[series.notna(), "date"]).dt.date
        return max(dates)

    @classmethod
    def _last_valid_date_for_symbol_iso(cls, frame: pd.DataFrame, symbol: str) -> str | None:
        last_valid = cls._last_valid_date_for_symbol(frame, symbol)
        return last_valid.isoformat() if last_valid is not None else None


def load_sector_price_history(
    *,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return SectorPriceStore(
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        symbols=symbols,
    ).load_offline()


def sync_sector_history(
    *,
    mode: str,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    target_end_date: date | str | None = None,
    bootstrap_start_date: date | str | None = None,
    history_years: int | None = None,
    history_days: int | None = None,
    overlap_trading_days: int = DEFAULT_OVERLAP_TRADING_DAYS,
    repair_start_date: date | str | None = None,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    readonly: bool = True,
    timeout_seconds: float = 10.0,
    market_data_type: int = 1,
    preferred_what_to_show: str = "ADJUSTED_LAST",
    allow_trades_fallback: bool = True,
    force: bool = False,
    data_pipe: IBKRDataPipe | None = None,
) -> SectorHistorySyncResult | pd.DataFrame | SectorPriceStoreValidation:
    store = SectorPriceStore(
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        symbols=symbols,
    )
    updater = IBKRHistoricalStoreUpdater(
        store=store,
        data_pipe=data_pipe,
        connection=IBKRConnectionConfig(
            host=host,
            port=int(port),
            client_id=int(client_id),
            readonly=bool(readonly),
            timeout_seconds=float(timeout_seconds),
            market_data_type=int(market_data_type),
        ),
        symbols=symbols,
        preferred_what_to_show=preferred_what_to_show,
        allow_trades_fallback=allow_trades_fallback,
        overlap_trading_days=overlap_trading_days,
    )
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "bootstrap":
        return updater.bootstrap(
            start_date=bootstrap_start_date,
            history_years=history_years,
            history_days=history_days,
            target_end_date=target_end_date,
            force=force,
        )
    if normalized_mode == "update":
        return updater.update(
            target_end_date=target_end_date,
            repair_start_date=repair_start_date,
            bootstrap_start_date=bootstrap_start_date,
        )
    if normalized_mode == "offline":
        return updater.load_offline()
    if normalized_mode == "validate":
        return updater.validate()
    raise ValueError(f"Unsupported sector history mode: {mode}")
