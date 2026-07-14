from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from agentic_vol_regime_app.data.ibkr_client import (
    IBKRDailyBar,
    IBKRDailyHistoryRequest,
    IBKRDailyHistoryResult,
)
from agentic_vol_regime_app.data.sector_history_gcs import (
    DEFAULT_GCS_PREFIX,
    GCSPublishResult,
    StorageManifestConflictError,
    StorageObjectMetadata,
    publish_sector_store_to_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import (
    SectorPriceStore,
    sync_sector_history,
)
from agentic_vol_regime_app.data.sector_history_update_publish import (
    SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES,
    update_and_publish_sector_history,
)
from agentic_vol_regime_app.data import (
    sector_history_update_publish as orchestration_module,
)
from agentic_vol_regime_app.data import sector_history_cli as cli_module


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
    skip_dates = skip_dates or set()
    history: dict[date, float] = {}
    for index, session_date in enumerate(_business_days(start, periods)):
        if session_date in skip_dates:
            continue
        history[session_date] = base + (index * step)
    return history


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


def _write_store(
    store: SectorPriceStore,
    frame: pd.DataFrame,
    *,
    actual_what_to_show_by_symbol: dict[str, str] | None = None,
) -> None:
    per_symbol_fetch = {}
    actual_map = actual_what_to_show_by_symbol or {}
    for symbol in store.symbols:
        actual = actual_map.get(symbol)
        per_symbol_fetch[symbol] = type(
            "FetchResult",
            (),
            {
                "actual_what_to_show": actual,
                "status": "bootstrap",
                "requested_start": None,
                "requested_end": None,
                "received": 0,
                "inserted": 0,
                "revised": 0,
            },
        )()
    metadata = store.build_metadata(
        frame=frame,
        mode="bootstrap",
        requested_settings={
            "bar_size": "1 day",
            "preferred_what_to_show": "ADJUSTED_LAST",
            "use_rth": True,
        },
        per_symbol_fetch=per_symbol_fetch,
        warnings=[],
    )
    store.write_authoritative(frame=frame, metadata=metadata)


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


@dataclass
class _StoredObject:
    data: bytes
    generation: int
    content_type: str


class FakeStorageClient:
    def __init__(self, *, bucket_exists: bool = True) -> None:
        self.bucket_exists_value = bucket_exists
        self.objects: dict[tuple[str, str], _StoredObject] = {}
        self.upload_calls: list[dict[str, object]] = []
        self.download_calls: list[tuple[str, str]] = []
        self.fail_uploads: set[str] = set()
        self.corrupt_downloads: dict[str, bytes] = {}
        self.force_manifest_conflict = False

    def bucket_exists(self, bucket: str) -> bool:
        return self.bucket_exists_value

    def get_object_metadata(
        self, bucket: str, object_name: str
    ) -> StorageObjectMetadata | None:
        stored = self.objects.get((bucket, object_name))
        if stored is None:
            return None
        return StorageObjectMetadata(
            bucket=bucket,
            object_name=object_name,
            generation=str(stored.generation),
            size_bytes=len(stored.data),
        )

    def upload_bytes(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        if_generation_match: int | str | None,
        content_type: str,
    ) -> StorageObjectMetadata:
        self.upload_calls.append(
            {
                "bucket": bucket,
                "object_name": object_name,
                "if_generation_match": if_generation_match,
                "content_type": content_type,
            }
        )
        if object_name in self.fail_uploads:
            raise RuntimeError(f"synthetic upload failure for {object_name}")
        if self.force_manifest_conflict and object_name.endswith(
            "/manifests/latest.json"
        ):
            raise StorageManifestConflictError("synthetic manifest conflict")
        key = (bucket, object_name)
        existing = self.objects.get(key)
        expected = None if if_generation_match is None else int(if_generation_match)
        if expected == 0 and existing is not None:
            raise StorageManifestConflictError("object already exists")
        if expected not in (None, 0):
            if existing is None or existing.generation != expected:
                raise StorageManifestConflictError("generation precondition failed")
        next_generation = 1 if existing is None else existing.generation + 1
        self.objects[key] = _StoredObject(
            data=bytes(data), generation=next_generation, content_type=content_type
        )
        return self.get_object_metadata(bucket, object_name)  # type: ignore[return-value]

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        self.download_calls.append((bucket, object_name))
        if object_name in self.corrupt_downloads:
            return self.corrupt_downloads[object_name]
        stored = self.objects.get((bucket, object_name))
        if stored is None:
            raise FileNotFoundError(object_name)
        return bytes(stored.data)


def _partial_store_with_histories(
    tmp_path: Path,
    histories: dict[str, dict[date, float]],
    *,
    periods: int,
    actual_what_to_show_by_symbol: dict[str, str] | None = None,
) -> SectorPriceStore:
    store = _build_store(tmp_path)
    partial_histories = {
        symbol: dict(list(values.items())[:periods])
        for symbol, values in histories.items()
    }
    frame = _make_frame_from_histories(partial_histories)
    _write_store(
        store, frame, actual_what_to_show_by_symbol=actual_what_to_show_by_symbol
    )
    return store


def _current_store_with_histories(
    tmp_path: Path,
    histories: dict[str, dict[date, float]],
    *,
    actual_what_to_show_by_symbol: dict[str, str] | None = None,
) -> SectorPriceStore:
    store = _build_store(tmp_path)
    frame = _make_frame_from_histories(histories)
    _write_store(
        store, frame, actual_what_to_show_by_symbol=actual_what_to_show_by_symbol
    )
    return store


def test_update_and_publish_calls_publish_then_verify_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    client = FakeDailyBarClient(histories)
    calls: list[str] = []

    def _publish(**kwargs):
        calls.append("publish")
        frame = store.load_offline()
        assert pd.to_datetime(frame["date"]).max().date().isoformat() == "2026-07-13"
        return GCSPublishResult(
            status="published",
            dataset_id="dataset-1",
            market_data_as_of="2026-07-13",
            bucket=str(kwargs["bucket"]),
            prefix=str(kwargs["prefix"]),
            parquet_uri="gs://test-bucket/market-manifold/datasets/dataset-1/sector_prices_daily.parquet",
            metadata_uri="gs://test-bucket/market-manifold/datasets/dataset-1/metadata.json",
            manifest_uri="gs://test-bucket/market-manifold/manifests/latest.json",
            parquet_sha256="abc123",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
        )

    def _verify(**_kwargs):
        calls.append("verify")
        return GCSPublishResult(
            status="verified",
            dataset_id="dataset-1",
            market_data_as_of="2026-07-13",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/market-manifold/datasets/dataset-1/sector_prices_daily.parquet",
            metadata_uri="gs://test-bucket/market-manifold/datasets/dataset-1/metadata.json",
            manifest_uri="gs://test-bucket/market-manifold/manifests/latest.json",
            parquet_sha256="abc123",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
            verified=True,
        )

    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(orchestration_module, "verify_sector_store_in_gcs", _verify)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=client,
    )

    assert result.status == "published"
    assert calls == ["publish", "verify"]


def test_publish_is_not_called_if_update_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _build_store(tmp_path)
    frame = pd.DataFrame(
        [{"date": pd.Timestamp("2026-07-10"), "XLK": 1.0, "XLE": 2.0, "SPY": 3.0}]
    )
    _write_store(store, frame)
    called = False

    def _publish(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("publish should not be called")

    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(
        orchestration_module,
        "sync_sector_history",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
    )

    assert result.status == "update_failed"
    assert called is False


def test_publish_is_not_called_if_local_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 5, base=100.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)
    called = False

    def _publish(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("publish should not be called")

    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(
        orchestration_module,
        "_validate_local_metadata",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad metadata")),
    )

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-06",
        data_pipe=FakeDailyBarClient(histories),
    )

    assert result.status == "local_validation_failed"
    assert called is False


def test_lagging_required_symbol_blocks_publication_and_reports_dates(
    tmp_path: Path,
) -> None:
    target = "2026-07-13"
    histories = {
        "XLK": _make_symbol_history("2026-06-30", 10, base=100.0),
        "XLE": _make_symbol_history(
            "2026-06-30", 10, base=200.0, skip_dates={date.fromisoformat(target)}
        ),
        "SPY": _make_symbol_history("2026-06-30", 10, base=300.0),
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    client = FakeDailyBarClient(histories)
    storage = FakeStorageClient()

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date=target,
        data_pipe=client,
        storage_client=storage,
    )

    assert result.status == "update_incomplete_not_published"
    assert result.gcs_publish.attempted is False
    assert (
        result.publication_readiness.lagging_symbols["XLE"]["expected_last_date"]
        == target
    )
    assert (
        result.publication_readiness.lagging_symbols["XLE"]["actual_last_date"]
        == "2026-07-10"
    )


def test_already_current_store_makes_zero_ibkr_requests_and_can_publish(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=150.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)
    client = FakeDailyBarClient(histories)
    storage = FakeStorageClient()

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=client,
        storage_client=storage,
    )

    assert result.status == "published"
    assert result.local_update.ibkr_request_count == 0
    assert client.requests == []


def test_already_current_publish_can_be_skipped(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=160.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        publish_if_already_current=False,
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    assert result.status == "already_current_publish_skipped"
    assert result.gcs_publish.attempted is False


def test_repeated_combined_runs_are_idempotent(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=170.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    storage = FakeStorageClient()

    first = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=storage,
    )
    second = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=storage,
    )

    assert first.status == "published"
    assert second.status == "already_published"
    assert second.local_update.ibkr_request_count == 0


def test_local_update_success_plus_gcs_failure_retains_local_update_and_manifest(
    tmp_path: Path,
) -> None:
    base_histories = {
        symbol: _make_symbol_history("2026-06-30", 9, base=180.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    updated_histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=180.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, base_histories)
    storage = FakeStorageClient()

    publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    manifest_key = ("test-bucket", f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")
    before_manifest = bytes(storage.objects[manifest_key].data)
    storage.fail_uploads.add(f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(updated_histories),
        storage_client=storage,
    )

    assert result.status == "local_updated_cloud_publish_failed"
    assert store.summarize_existing_store()["last_date"] == "2026-07-13"
    assert storage.objects[manifest_key].data == before_manifest


def test_verify_failure_returns_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=190.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)

    def _verify(**_kwargs):
        raise ValueError("synthetic verify failure")

    monkeypatch.setattr(orchestration_module, "verify_sector_store_in_gcs", _verify)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    assert result.status == "verification_failed"
    assert result.gcs_verify.attempted is True
    assert result.gcs_verify.verified is False


def test_dataset_id_mismatch_causes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=200.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)

    def _verify(**_kwargs):
        return GCSPublishResult(
            status="verified",
            dataset_id="different-id",
            market_data_as_of="2026-07-13",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="same-sha",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
            verified=True,
        )

    def _publish(**_kwargs):
        return GCSPublishResult(
            status="published",
            dataset_id="expected-id",
            market_data_as_of="2026-07-13",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="same-sha",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
        )

    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(orchestration_module, "verify_sector_store_in_gcs", _verify)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
    )

    assert result.status == "verification_failed"
    assert "dataset_id mismatch" in result.errors[0]


def test_checksum_mismatch_causes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=210.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)

    def _verify(**_kwargs):
        return GCSPublishResult(
            status="verified",
            dataset_id="expected-id",
            market_data_as_of="2026-07-13",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="different-sha",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
            verified=True,
        )

    def _publish(**_kwargs):
        return GCSPublishResult(
            status="published",
            dataset_id="expected-id",
            market_data_as_of="2026-07-13",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="same-sha",
            row_count=10,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-06-30",
            last_date="2026-07-13",
        )

    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(orchestration_module, "verify_sector_store_in_gcs", _verify)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
    )

    assert result.status == "verification_failed"
    assert "checksum mismatch" in result.errors[0]


def test_default_trades_fallback_warns_and_may_publish(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=220.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(
        tmp_path, histories, actual_what_to_show_by_symbol={"XLE": "TRADES"}
    )

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    assert result.status == "published"
    assert any(
        "XLE publication is using TRADES" in warning for warning in result.warnings
    )


def test_require_adjusted_last_blocks_trades(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=230.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(
        tmp_path, histories, actual_what_to_show_by_symbol={"XLE": "TRADES"}
    )

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        require_adjusted_last=True,
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    assert result.status == "update_incomplete_not_published"
    assert any(
        "strict adjusted-only mode" in reason
        for reason in result.publication_readiness.blocked_reasons
    )


def test_require_adjusted_last_blocks_mixed(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=240.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(
        tmp_path, histories, actual_what_to_show_by_symbol={"XLE": "MIXED"}
    )

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        require_adjusted_last=True,
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    assert result.status == "update_incomplete_not_published"
    assert any(
        "MIXED" in reason for reason in result.publication_readiness.blocked_reasons
    )


def test_dry_run_updates_locally_and_skips_gcs_writes_and_verify(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=250.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    storage = FakeStorageClient()

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        dry_run_publish=True,
        data_pipe=FakeDailyBarClient(histories),
        storage_client=storage,
    )

    assert result.status == "dry_run_completed"
    assert store.summarize_existing_store()["last_date"] == "2026-07-13"
    assert storage.upload_calls == []
    assert result.gcs_verify.attempted is False


def test_combined_result_contains_request_count_and_is_json_safe_and_secret_free(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=260.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=FakeStorageClient(),
    )

    payload = result.to_dict()
    json.dumps(payload, allow_nan=False)

    assert payload["local_update"]["ibkr_request_count"] == 3
    serialized = json.dumps(payload)
    assert "token" not in serialized.lower()
    assert "secret" not in serialized.lower()
    assert "authorization" not in serialized.lower()


def test_missing_bucket_fails_before_ibkr_update(tmp_path: Path) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=270.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    client = FakeDailyBarClient(histories)

    result = update_and_publish_sector_history(
        bucket="",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=client,
        storage_client=FakeStorageClient(),
    )

    assert result.status == "invalid_configuration"
    assert client.requests == []


def test_missing_local_store_fails_with_bootstrap_instruction(tmp_path: Path) -> None:
    store = _build_store(tmp_path)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-10",
    )

    assert result.status == "invalid_configuration"
    assert "Bootstrap" in result.errors[0]


def test_orchestration_reuses_existing_update_publish_and_verify_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _build_store(tmp_path)
    frame = pd.DataFrame(
        [{"date": pd.Timestamp("2026-07-10"), "XLK": 1.0, "XLE": 2.0, "SPY": 3.0}]
    )
    _write_store(store, frame)
    calls: list[str] = []

    def _sync(**_kwargs):
        calls.append("sync")
        return sync_sector_history(
            mode="update",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(TEST_SYMBOLS),
            target_end_date="2026-07-10",
            data_pipe=FakeDailyBarClient(
                {
                    symbol: {date.fromisoformat("2026-07-10"): 10.0}
                    for symbol in TEST_SYMBOLS
                }
            ),
        )

    def _publish(**_kwargs):
        calls.append("publish")
        return GCSPublishResult(
            status="published",
            dataset_id="dataset-1",
            market_data_as_of="2026-07-10",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="same-sha",
            row_count=1,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-07-10",
            last_date="2026-07-10",
        )

    def _verify(**_kwargs):
        calls.append("verify")
        return GCSPublishResult(
            status="verified",
            dataset_id="dataset-1",
            market_data_as_of="2026-07-10",
            bucket="test-bucket",
            prefix=DEFAULT_GCS_PREFIX,
            parquet_uri="gs://test-bucket/x",
            metadata_uri="gs://test-bucket/y",
            manifest_uri="gs://test-bucket/z",
            parquet_sha256="same-sha",
            row_count=1,
            symbols=list(TEST_SYMBOLS),
            first_date="2026-07-10",
            last_date="2026-07-10",
            verified=True,
        )

    monkeypatch.setattr(orchestration_module, "sync_sector_history", _sync)
    monkeypatch.setattr(orchestration_module, "publish_sector_store_to_gcs", _publish)
    monkeypatch.setattr(orchestration_module, "verify_sector_store_in_gcs", _verify)

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-10",
    )

    assert result.status == "published"
    assert calls == ["sync", "publish", "verify"]


def test_regression_delta_update_only_requests_bounded_overlap_before_publish(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=280.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _partial_store_with_histories(tmp_path, histories, periods=9)
    client = FakeDailyBarClient(histories)
    storage = FakeStorageClient()

    result = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        overlap_trading_days=5,
        data_pipe=client,
        storage_client=storage,
    )

    object_names = [str(call["object_name"]) for call in storage.upload_calls]
    assert result.status == "published"
    assert client.requests
    assert (
        max(int(request["duration_calendar_days"]) for request in client.requests) < 30
    )
    assert object_names[-1].endswith("/manifests/latest.json")


def test_regression_already_current_rerun_is_verified_idempotently(
    tmp_path: Path,
) -> None:
    histories = {
        symbol: _make_symbol_history("2026-06-30", 10, base=290.0 + index)
        for index, symbol in enumerate(TEST_SYMBOLS)
    }
    store = _current_store_with_histories(tmp_path, histories)
    storage = FakeStorageClient()

    first = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=storage,
    )
    second = update_and_publish_sector_history(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(TEST_SYMBOLS),
        target_end_date="2026-07-13",
        data_pipe=FakeDailyBarClient(histories),
        storage_client=storage,
    )

    assert first.status == "published"
    assert second.local_update.ibkr_request_count == 0
    assert second.status in {"published", "already_published"}
    assert second.gcs_verify.verified is True


def test_cli_exit_codes_follow_combined_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = orchestration_module._empty_result(  # type: ignore[attr-defined]
        status="update_failed",
        target_completed_session="2026-07-13",
        started_at="2026-07-13T00:00:00Z",
        errors=["boom"],
    )
    monkeypatch.setattr(
        cli_module, "update_and_publish_sector_history", lambda **_kwargs: result
    )
    monkeypatch.setattr(
        cli_module.sys,
        "argv",
        [
            "sector_history_cli.py",
            "update-and-publish-gcs",
            "--bucket",
            "test-bucket",
            "--output",
            str(tmp_path / "x.parquet"),
            "--metadata-output",
            str(tmp_path / "x.metadata.json"),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 1
    assert "update_failed" in capsys.readouterr().out


def test_success_status_set_is_used_for_zero_exit() -> None:
    assert "published" in SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES
    assert "already_published" in SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES
    assert "already_current_publish_skipped" in SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES
    assert "dry_run_completed" in SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES
