from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import pytest

from agentic_vol_regime_app.data.ibkr_client import (
    IBKRDailyBar,
    IBKRDailyHistoryRequest,
    IBKRDailyHistoryResult,
)
from agentic_vol_regime_app.data.sector_history_store import (
    IBKRHistoricalStoreUpdater,
    SectorPriceStore,
    _frame_content_sha256,
    load_sector_price_history,
    resolve_target_completed_session,
    sync_sector_history,
)


TEST_SYMBOLS = ("XLK", "XLE", "SPY")


def _business_days(start: str, periods: int) -> list[date]:
    return list(pd.bdate_range(start=start, periods=periods).date)


def _make_symbol_history(
    start: str,
    periods: int,
    *,
    base: float,
    step: float = 1.0,
    skip_dates: set[date] | None = None,
) -> dict[date, float]:
    history: dict[date, float] = {}
    skip_dates = skip_dates or set()
    for index, day in enumerate(_business_days(start, periods)):
        if day in skip_dates:
            continue
        history[day] = base + (index * step)
    return history


class FakeDailyBarClient:
    def __init__(
        self,
        history_by_symbol: dict[str, dict[date, float]],
        *,
        actual_what_to_show_by_symbol: dict[str, str] | None = None,
        fail_on_request_number: int | None = None,
    ) -> None:
        self.history_by_symbol = history_by_symbol
        self.actual_what_to_show_by_symbol = actual_what_to_show_by_symbol or {}
        self.fail_on_request_number = fail_on_request_number
        self.requests: list[dict[str, object]] = []

    def request_daily_bars(
        self, request: IBKRDailyHistoryRequest
    ) -> IBKRDailyHistoryResult:
        request_number = len(self.requests) + 1
        if (
            self.fail_on_request_number is not None
            and request_number == self.fail_on_request_number
        ):
            raise RuntimeError("synthetic IBKR request failure")
        end_date = pd.Timestamp(
            request.end_date or max(self.history_by_symbol[request.symbol].keys())
        )
        start_date = end_date - pd.Timedelta(
            days=max(int(request.duration_calendar_days), 1) - 1
        )
        self.requests.append(
            {
                "symbol": request.symbol,
                "end_date": end_date.date(),
                "duration_calendar_days": int(request.duration_calendar_days),
                "estimated_start": start_date.date()
                if hasattr(start_date, "date")
                else start_date,
            }
        )
        actual_what = self.actual_what_to_show_by_symbol.get(
            request.symbol, request.preferred_what_to_show
        )
        bars = []
        for session_date, close in sorted(
            self.history_by_symbol.get(request.symbol, {}).items()
        ):
            if start_date <= pd.Timestamp(session_date) <= end_date:
                bars.append(
                    IBKRDailyBar(
                        symbol=request.symbol,
                        session_date=session_date,
                        close=float(close),
                        actual_what_to_show=actual_what,
                        source="IBKR",
                    )
                )
        return IBKRDailyHistoryResult(
            bars=tuple(bars), actual_what_to_show=actual_what, warnings=()
        )


def _store_paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / "sector_prices_daily.parquet",
        tmp_path / "sector_prices_daily.metadata.json",
    )


def _build_store(
    tmp_path: Path, symbols: tuple[str, ...] = TEST_SYMBOLS
) -> SectorPriceStore:
    parquet_path, metadata_path = _store_paths(tmp_path)
    return SectorPriceStore(
        parquet_path=parquet_path, metadata_path=metadata_path, symbols=symbols
    )


def _write_existing_store(store: SectorPriceStore, frame: pd.DataFrame) -> None:
    metadata = store.build_metadata(
        frame=frame,
        mode="bootstrap",
        requested_settings={
            "bar_size": "1 day",
            "preferred_what_to_show": "ADJUSTED_LAST",
            "use_rth": True,
        },
        per_symbol_fetch={},
        warnings=[],
    )
    store.write_authoritative(frame=frame, metadata=metadata)


def _make_frame_from_histories(
    histories: dict[str, dict[date, float]], symbols: tuple[str, ...] = TEST_SYMBOLS
) -> pd.DataFrame:
    all_dates = sorted(
        {day for symbol_history in histories.values() for day in symbol_history.keys()}
    )
    rows = []
    for day in all_dates:
        row: dict[str, object] = {"date": pd.Timestamp(day)}
        for symbol in symbols:
            row[symbol] = histories.get(symbol, {}).get(day)
        rows.append(row)
    return pd.DataFrame(rows, columns=["date", *symbols])


def _make_updater(
    tmp_path: Path,
    histories: dict[str, dict[date, float]],
    *,
    actual_what_to_show_by_symbol: dict[str, str] | None = None,
    overlap_trading_days: int = 5,
    repair_horizon_trading_days: int = 10,
    chunk_calendar_days: int = 365,
) -> tuple[SectorPriceStore, IBKRHistoricalStoreUpdater, FakeDailyBarClient]:
    store = _build_store(tmp_path)
    client = FakeDailyBarClient(
        histories,
        actual_what_to_show_by_symbol=actual_what_to_show_by_symbol,
    )
    updater = IBKRHistoricalStoreUpdater(
        store=store,
        data_pipe=client,
        symbols=TEST_SYMBOLS,
        overlap_trading_days=overlap_trading_days,
        repair_horizon_trading_days=repair_horizon_trading_days,
        chunk_calendar_days=chunk_calendar_days,
    )
    return store, updater, client


def test_bootstrap_creates_dated_sorted_unique_parquet_store(tmp_path: Path) -> None:
    histories = {
        "XLK": _make_symbol_history("2026-06-02", 8, base=200.0),
        "XLE": _make_symbol_history("2026-06-02", 8, base=90.0),
        "SPY": _make_symbol_history("2026-06-02", 8, base=500.0),
    }
    store, updater, _client = _make_updater(tmp_path, histories)

    result = updater.bootstrap(start_date="2026-06-02", target_end_date="2026-06-11")
    frame = store.load_offline()

    assert result.mode == "bootstrap"
    assert list(frame.columns) == ["date", *TEST_SYMBOLS]
    assert frame["date"].is_monotonic_increasing
    assert not frame["date"].duplicated().any()
    assert frame["XLK"].iloc[0] == 200.0


def test_bootstrap_refuses_overwrite_without_force(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-02", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-02", target_end_date="2026-06-08")

    with pytest.raises(FileExistsError):
        updater.bootstrap(start_date="2026-06-02", target_end_date="2026-06-08")


def test_second_update_on_already_current_store_makes_zero_requests(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-02", 6, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-02", target_end_date="2026-06-09")
    client.requests.clear()

    result = updater.update(target_end_date="2026-06-09")

    assert result.ibkr_request_count == 0
    assert client.requests == []
    assert all(
        payload.status == "already_current" for payload in result.symbols.values()
    )


def test_update_requests_only_delta_plus_overlap(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories, overlap_trading_days=5)
    frame = _make_frame_from_histories(
        {symbol: dict(list(values.items())[:9]) for symbol, values in histories.items()}
    )
    _write_existing_store(store, frame)

    updater.update(target_end_date="2026-07-13")

    assert len(client.requests) == 3
    assert (
        max(int(request["duration_calendar_days"]) for request in client.requests) <= 20
    )


def test_different_per_symbol_watermarks_produce_different_request_ranges(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=150.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories, overlap_trading_days=2)
    full_frame = _make_frame_from_histories(histories)
    full_frame.loc[full_frame.index[-1], "XLE"] = None
    full_frame.loc[full_frame.index[-2:], "SPY"] = None
    _write_existing_store(store, full_frame)

    updater.update(target_end_date="2026-07-13")

    estimated_starts = {
        request["symbol"]: request["estimated_start"] for request in client.requests
    }
    assert estimated_starts["XLE"] != estimated_starts["SPY"]


def test_missing_stored_history_for_one_symbol_does_not_force_all_symbols_to_refetch(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=120.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories)
    frame = _make_frame_from_histories(histories)
    frame["XLE"] = pd.NA
    _write_existing_store(store, frame)

    result = updater.update(
        target_end_date="2026-07-13", bootstrap_start_date="2026-06-30"
    )

    assert result.symbols["XLK"].status == "already_current"
    assert result.symbols["SPY"].status == "already_current"
    assert result.symbols["XLE"].status == "updated"
    assert [request["symbol"] for request in client.requests] == ["XLE"]


def test_overlapping_returned_dates_are_deduplicated(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2025-01-02", 280, base=180.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(
        tmp_path, histories, chunk_calendar_days=180
    )

    updater.bootstrap(start_date="2025-01-02", target_end_date="2026-01-30")
    frame = store.load_offline()

    assert not frame["date"].duplicated().any()


def test_new_valid_overlapping_values_replace_stored_values(tmp_path: Path) -> None:
    histories = {
        "XLK": _make_symbol_history("2026-06-30", 7, base=200.0),
        "XLE": _make_symbol_history("2026-06-30", 7, base=90.0),
        "SPY": _make_symbol_history("2026-06-30", 7, base=500.0),
    }
    revised_day = _business_days("2026-06-30", 7)[-2]
    histories["XLK"][revised_day] = 999.0
    store, updater, _client = _make_updater(tmp_path, histories, overlap_trading_days=3)
    existing = _make_frame_from_histories(
        {symbol: dict(list(values.items())[:6]) for symbol, values in histories.items()}
    )
    existing.loc[existing["date"] == pd.Timestamp(revised_day), "XLK"] = 250.0
    _write_existing_store(store, existing)

    updater.update(target_end_date="2026-07-08")
    frame = store.load_offline()

    assert (
        float(frame.loc[frame["date"] == pd.Timestamp(revised_day), "XLK"].iloc[0])
        == 999.0
    )


def test_revised_value_count_is_accurate(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 7, base=300.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    revised_days = _business_days("2026-06-30", 7)[-3:-1]
    histories["SPY"][revised_days[0]] = 700.0
    histories["SPY"][revised_days[1]] = 701.0
    store, updater, _client = _make_updater(tmp_path, histories, overlap_trading_days=4)
    existing = _make_frame_from_histories(
        {symbol: dict(list(values.items())[:6]) for symbol, values in histories.items()}
    )
    existing.loc[existing["date"] == pd.Timestamp(revised_days[0]), "SPY"] = 600.0
    existing.loc[existing["date"] == pd.Timestamp(revised_days[1]), "SPY"] = 601.0
    _write_existing_store(store, existing)

    result = updater.update(target_end_date="2026-07-08")

    assert result.symbols["SPY"].revised == 2


def test_missing_price_remains_missing_and_is_never_forward_filled(
    tmp_path: Path,
) -> None:
    missing_day = _business_days("2026-06-30", 8)[4]
    histories = {
        "XLK": _make_symbol_history(
            "2026-06-30", 8, base=200.0, skip_dates={missing_day}
        ),
        "XLE": _make_symbol_history("2026-06-30", 8, base=90.0),
        "SPY": _make_symbol_history("2026-06-30", 8, base=500.0),
    }
    store, updater, _client = _make_updater(tmp_path, histories)

    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-09")
    frame = store.load_offline()

    row = frame.loc[frame["date"] == pd.Timestamp(missing_day)]
    assert pd.isna(row["XLK"].iloc[0])


def test_missing_bar_does_not_shift_dates_or_misalign_other_symbols(
    tmp_path: Path,
) -> None:
    missing_day = _business_days("2026-06-30", 8)[2]
    histories = {
        "XLK": _make_symbol_history(
            "2026-06-30", 8, base=200.0, skip_dates={missing_day}
        ),
        "XLE": _make_symbol_history("2026-06-30", 8, base=90.0),
        "SPY": _make_symbol_history("2026-06-30", 8, base=500.0),
    }
    store, updater, _client = _make_updater(tmp_path, histories)

    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-09")
    frame = store.load_offline()

    row = frame.loc[frame["date"] == pd.Timestamp(missing_day)].iloc[0]
    assert pd.isna(row["XLK"])
    assert row["XLE"] > 0
    assert row["SPY"] > 0


def test_offline_mode_makes_zero_ibkr_requests(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-06")
    client.requests.clear()

    frame = load_sector_price_history(
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
    )

    assert not frame.empty
    assert client.requests == []


def test_offline_mode_succeeds_when_ibkr_is_unavailable(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=110.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-06")

    class BrokenClient:
        def request_daily_bars(
            self, _request
        ):  # pragma: no cover - should never be called
            raise AssertionError("offline mode must not request IBKR bars")

    frame = sync_sector_history(
        mode="offline",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        data_pipe=BrokenClient(),
    )

    assert len(frame) == 5


def test_offline_mode_fails_clearly_when_store_does_not_exist(tmp_path: Path) -> None:
    store = _build_store(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load_offline()


def test_corrupt_duplicate_stored_dates_fail_validation(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    duplicate_day = pd.Timestamp("2026-07-01")
    frame = pd.DataFrame(
        [
            {"date": duplicate_day, "XLK": 1.0, "XLE": 2.0, "SPY": 3.0},
            {"date": duplicate_day, "XLK": 1.1, "XLE": 2.1, "SPY": 3.1},
        ]
    )

    with pytest.raises(ValueError):
        store.validate_frame(frame)


def test_zero_negative_and_infinity_values_are_rejected_while_nan_is_preserved(
    tmp_path: Path,
) -> None:
    store = _build_store(tmp_path)
    valid_frame = pd.DataFrame(
        [{"date": pd.Timestamp("2026-07-01"), "XLK": pd.NA, "XLE": 10.0, "SPY": 20.0}]
    )
    assert store.validate_frame(valid_frame).row_count == 1

    for bad_value in (0.0, -1.0, float("inf")):
        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-07-01"),
                    "XLK": bad_value,
                    "XLE": 10.0,
                    "SPY": 20.0,
                }
            ]
        )
        with pytest.raises(ValueError):
            store.validate_frame(frame)


def test_failed_update_leaves_prior_authoritative_store_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 8, base=200.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-08")
    before = store.load_offline().copy()

    def _fail_write(*, frame, metadata):
        raise RuntimeError("synthetic persistence failure")

    monkeypatch.setattr(store, "write_authoritative", _fail_write)
    with pytest.raises(RuntimeError):
        updater.update(target_end_date="2026-07-09")

    after = SectorPriceStore(
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
    ).load_offline()
    pd.testing.assert_frame_equal(before, after)


def test_metadata_checksum_matches_stored_parquet(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-06")

    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    frame = store.load_offline()

    assert metadata["content_sha256"] == _frame_content_sha256(frame)


def test_weekend_explicit_target_dates_normalize_consistently() -> None:
    assert resolve_target_completed_session(
        explicit_target_end_date="2026-07-12"
    ) == date(2026, 7, 10)


def test_running_during_incomplete_session_does_not_mark_today_completed() -> None:
    now = datetime(2026, 7, 10, 15, 59, tzinfo=timezone.utc)
    target = resolve_target_completed_session(
        now=now,
        market_timezone="America/New_York",
        market_close_cutoff=time(hour=16, minute=15),
    )

    assert target == date(2026, 7, 9)


def test_actual_what_to_show_is_recorded_accurately(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(
        tmp_path,
        histories,
        actual_what_to_show_by_symbol={
            "XLK": "ADJUSTED_LAST",
            "XLE": "ADJUSTED_LAST",
            "SPY": "ADJUSTED_LAST",
        },
    )

    result = updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-06")
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))

    assert result.symbols["XLK"].actual_what_to_show == "ADJUSTED_LAST"
    assert metadata["per_symbol"]["XLK"]["actual_what_to_show"] == "ADJUSTED_LAST"


def test_trades_fallback_generates_warning_and_is_not_labeled_adjusted(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, _client = _make_updater(
        tmp_path, histories, actual_what_to_show_by_symbol={"XLE": "TRADES"}
    )

    result = updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-06")
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))

    assert result.symbols["XLE"].actual_what_to_show == "TRADES"
    assert any("fell back to TRADES" in warning for warning in result.warnings)
    assert metadata["per_symbol"]["XLE"]["actual_what_to_show"] == "TRADES"


def test_long_bootstrap_chunking_preserves_dates_and_deduplicates_overlaps(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2025-01-02", 320, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories, chunk_calendar_days=120)

    updater.bootstrap(start_date="2025-01-02", target_end_date="2026-03-25")
    frame = store.load_offline()

    assert len(client.requests) > 3
    assert frame["date"].nunique() == len(frame)


def test_update_does_not_issue_full_history_request_regression(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=140.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories, overlap_trading_days=5)
    existing = _make_frame_from_histories(
        {symbol: dict(list(values.items())[:9]) for symbol, values in histories.items()}
    )
    _write_existing_store(store, existing)

    updater.update(target_end_date="2026-07-13", bootstrap_start_date="2020-01-01")

    assert client.requests
    assert (
        max(int(request["duration_calendar_days"]) for request in client.requests) < 30
    )


def test_repeating_same_update_is_idempotent(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=160.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(tmp_path, histories)
    updater.bootstrap(start_date="2026-06-30", target_end_date="2026-07-13")
    frame_before = store.load_offline().copy()
    client.requests.clear()

    first = updater.update(target_end_date="2026-07-13")
    second = updater.update(target_end_date="2026-07-13")
    frame_after = store.load_offline()

    assert first.ibkr_request_count == 0
    assert second.ibkr_request_count == 0
    pd.testing.assert_frame_equal(frame_before, frame_after)


def test_recent_internal_gap_is_repaired_with_bounded_request(tmp_path: Path) -> None:
    gap_day = _business_days("2026-06-30", 8)[-2]
    histories = {
        symbol: _make_symbol_history("2026-06-30", 8, base=170.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store, updater, client = _make_updater(
        tmp_path, histories, repair_horizon_trading_days=5
    )
    existing = _make_frame_from_histories(histories)
    existing.loc[existing["date"] == pd.Timestamp(gap_day), "XLK"] = pd.NA
    _write_existing_store(store, existing)

    result = updater.update(target_end_date="2026-07-09")
    frame = store.load_offline()

    assert result.symbols["XLK"].status == "updated"
    assert pd.notna(frame.loc[frame["date"] == pd.Timestamp(gap_day), "XLK"].iloc[0])
