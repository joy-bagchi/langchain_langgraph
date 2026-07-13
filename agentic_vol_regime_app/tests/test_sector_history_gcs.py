from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from agentic_vol_regime_app.data.sector_history_gcs import (
    DEFAULT_GCS_PREFIX,
    BucketMissingError,
    GCSPublishResult,
    StorageManifestConflictError,
    StorageObjectConflictError,
    StorageObjectMetadata,
    _dataset_id,
    publish_sector_store_to_gcs,
    verify_sector_store_in_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import SectorPriceStore


TEST_SYMBOLS = ("XLK", "XLE", "SPY")


def _store_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "sector_prices_daily.parquet", tmp_path / "sector_prices_daily.metadata.json"


def _build_store(tmp_path: Path) -> SectorPriceStore:
    parquet_path, metadata_path = _store_paths(tmp_path)
    return SectorPriceStore(parquet_path=parquet_path, metadata_path=metadata_path, symbols=TEST_SYMBOLS)


def _build_frame() -> pd.DataFrame:
    dates = pd.bdate_range(start="2026-07-06", periods=5)
    rows = []
    for index, session_date in enumerate(dates):
        rows.append(
            {
                "date": session_date,
                "XLK": 200.0 + index,
                "XLE": 90.0 + index,
                "SPY": 500.0 + index,
            }
        )
    return pd.DataFrame(rows, columns=["date", *TEST_SYMBOLS])


def _write_store(tmp_path: Path) -> SectorPriceStore:
    store = _build_store(tmp_path)
    frame = _build_frame()
    metadata = store.build_metadata(
        frame=frame,
        mode="bootstrap",
        requested_settings={"bar_size": "1 day", "preferred_what_to_show": "ADJUSTED_LAST", "use_rth": True},
        per_symbol_fetch={},
        warnings=[],
    )
    store.write_authoritative(frame=frame, metadata=metadata)
    return store


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

    def get_object_metadata(self, bucket: str, object_name: str) -> StorageObjectMetadata | None:
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
        if self.force_manifest_conflict and object_name.endswith("/manifests/latest.json"):
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
        self.objects[key] = _StoredObject(data=bytes(data), generation=next_generation, content_type=content_type)
        return self.get_object_metadata(bucket, object_name)  # type: ignore[return-value]

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        self.download_calls.append((bucket, object_name))
        if object_name in self.corrupt_downloads:
            return self.corrupt_downloads[object_name]
        stored = self.objects.get((bucket, object_name))
        if stored is None:
            raise FileNotFoundError(object_name)
        return bytes(stored.data)


def test_valid_local_store_produces_deterministic_dataset_id(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    result = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=FakeStorageClient(),
    )

    assert result.dataset_id == _dataset_id(market_data_as_of=result.last_date, parquet_sha256=result.parquet_sha256)


def test_dry_run_performs_zero_gcs_writes(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()

    result = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    assert result.status == "dry_run"
    assert storage.upload_calls == []


def test_publishing_uploads_parquet_and_metadata_before_manifest(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()

    publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    object_names = [str(call["object_name"]) for call in storage.upload_calls]
    assert object_names[0].endswith("/sector_prices_daily.parquet")
    assert object_names[1].endswith("/metadata.json")
    assert object_names[2].endswith("/manifests/latest.json")


def test_manifest_is_not_written_if_parquet_upload_fails(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    dataset = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    parquet_object = dataset.parquet_uri.removeprefix("gs://test-bucket/")
    storage.fail_uploads.add(parquet_object)

    with pytest.raises(RuntimeError):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )

    assert not any(call["object_name"] == f"{DEFAULT_GCS_PREFIX}/manifests/latest.json" for call in storage.upload_calls)


def test_manifest_is_not_written_if_metadata_upload_fails(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    dataset = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    metadata_object = dataset.metadata_uri.removeprefix("gs://test-bucket/")
    storage.fail_uploads.add(metadata_object)

    with pytest.raises(RuntimeError):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )

    assert not any(call["object_name"] == f"{DEFAULT_GCS_PREFIX}/manifests/latest.json" for call in storage.upload_calls)


def test_manifest_is_not_written_if_checksum_verification_fails(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    dataset = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    parquet_object = dataset.parquet_uri.removeprefix("gs://test-bucket/")
    storage.corrupt_downloads[parquet_object] = b"corrupt"

    with pytest.raises(ValueError, match="SHA-256"):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )

    assert (("test-bucket", f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")) not in storage.objects


def test_uploaded_checksums_match_local_checksums(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()

    result = publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    manifest = json.loads(storage.objects[("test-bucket", f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")].data.decode("utf-8"))
    assert manifest["parquet"]["sha256"] == result.parquet_sha256
    assert manifest["metadata"]["sha256"]


def test_existing_identical_immutable_objects_are_reused(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    first = publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    upload_count = len(storage.upload_calls)

    second = publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    assert first.dataset_id == second.dataset_id
    assert second.status == "already_published"
    assert len(storage.upload_calls) == upload_count
    assert second.reused_objects


def test_conflicting_immutable_object_fails(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    dry_run = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    parquet_object = dry_run.parquet_uri.removeprefix("gs://test-bucket/")
    storage.objects[("test-bucket", parquet_object)] = _StoredObject(
        data=b"different",
        generation=1,
        content_type="application/octet-stream",
    )

    with pytest.raises(StorageObjectConflictError):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )


def test_existing_latest_manifest_is_unchanged_after_failed_publication(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    manifest_key = ("test-bucket", f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")
    before = bytes(storage.objects[manifest_key].data)

    frame = store.load_offline().copy()
    frame.loc[frame.index[-1], "SPY"] = 777.0
    metadata = store.build_metadata(
        frame=frame,
        mode="update",
        requested_settings={"bar_size": "1 day", "preferred_what_to_show": "ADJUSTED_LAST", "use_rth": True},
        per_symbol_fetch={},
        warnings=[],
    )
    store.write_authoritative(frame=frame, metadata=metadata)

    dry_run = publish_sector_store_to_gcs(
        bucket="test-bucket",
        dry_run=True,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    metadata_object = dry_run.metadata_uri.removeprefix("gs://test-bucket/")
    storage.fail_uploads.add(metadata_object)

    with pytest.raises(RuntimeError):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )

    assert storage.objects[manifest_key].data == before


def test_manifest_update_uses_generation_precondition_and_conflict_fails_clearly(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    store_two = _build_store(tmp_path)
    frame = store_two.load_offline().copy()
    frame.loc[frame.index[-1], "XLK"] = 999.0
    metadata = store_two.build_metadata(
        frame=frame,
        mode="update",
        requested_settings={"bar_size": "1 day", "preferred_what_to_show": "ADJUSTED_LAST", "use_rth": True},
        per_symbol_fetch={},
        warnings=[],
    )
    store_two.write_authoritative(frame=frame, metadata=metadata)
    storage.force_manifest_conflict = True

    with pytest.raises(StorageManifestConflictError, match="changed concurrently"):
        publish_sector_store_to_gcs(
            bucket="test-bucket",
            parquet_path=store_two.parquet_path,
            metadata_path=store_two.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=storage,
        )

    manifest_calls = [call for call in storage.upload_calls if str(call["object_name"]).endswith("/manifests/latest.json")]
    assert manifest_calls[-1]["if_generation_match"] == 1


def test_verify_gcs_validates_referenced_parquet(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()
    publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )

    result = verify_sector_store_in_gcs(bucket="test-bucket", storage_client=storage)

    assert result.status == "verified"
    assert result.verified is True
    assert result.last_date == "2026-07-10"


def test_missing_bucket_produces_useful_error(tmp_path: Path) -> None:
    store = _write_store(tmp_path)

    with pytest.raises(BucketMissingError, match="Create it explicitly"):
        publish_sector_store_to_gcs(
            bucket="missing-bucket",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=TEST_SYMBOLS,
            storage_client=FakeStorageClient(bucket_exists=False),
        )


def test_publisher_makes_zero_ibkr_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()

    def _boom(*_args, **_kwargs):
        raise AssertionError("IBKRLiveClient must not be instantiated during GCS publish/verify")

    monkeypatch.setattr("agentic_vol_regime_app.data.ibkr_client.IBKRLiveClient", _boom)

    publish_result = publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    verify_result = verify_sector_store_in_gcs(bucket="test-bucket", storage_client=storage)

    assert publish_result.ibkr_request_count == 0
    assert verify_result.ibkr_request_count == 0


def test_result_and_manifest_are_json_safe(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    storage = FakeStorageClient()

    result = publish_sector_store_to_gcs(
        bucket="test-bucket",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=TEST_SYMBOLS,
        storage_client=storage,
    )
    manifest_bytes = storage.objects[("test-bucket", f"{DEFAULT_GCS_PREFIX}/manifests/latest.json")].data

    json.dumps(result.to_dict(), allow_nan=False)
    json.dumps(json.loads(manifest_bytes.decode("utf-8")), allow_nan=False)

    assert isinstance(result, GCSPublishResult)
