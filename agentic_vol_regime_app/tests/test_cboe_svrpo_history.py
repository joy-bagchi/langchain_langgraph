from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_vol_regime_app.data.cboe_svrpo_history import (
    SVRPO_SOURCE_URL,
    download_svrpo_csv,
    parse_svrpo_csv,
    sync_and_publish_svrpo_history,
)
from agentic_vol_regime_app.data.sector_history_gcs import (
    StorageManifestConflictError,
    StorageObjectMetadata,
)


CSV_V1 = b"DATE,SVRPO\n12/30/2005,100.0\n09/14/2026,120.5\n"
CSV_REVISED = b"DATE,SVRPO\n12/30/2005,101.0\n09/14/2026,120.5\n"
NOW = datetime(2026, 9, 15, 18, 0, tzinfo=timezone.utc)


@dataclass
class _Object:
    data: bytes
    generation: int


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], _Object] = {}
        self.operations: list[str] = []
        self.fail_next_upload = False

    def bucket_exists(self, bucket: str) -> bool:
        self.operations.append("bucket_exists")
        return True

    def get_object_metadata(
        self, bucket: str, object_name: str
    ) -> StorageObjectMetadata | None:
        self.operations.append(f"metadata:{object_name}")
        value = self.objects.get((bucket, object_name))
        if value is None:
            return None
        return StorageObjectMetadata(
            bucket, object_name, str(value.generation), len(value.data)
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
        self.operations.append(f"upload:{object_name}")
        if self.fail_next_upload:
            self.fail_next_upload = False
            raise RuntimeError("synthetic upload failure")
        key = (bucket, object_name)
        existing = self.objects.get(key)
        expected = None if if_generation_match is None else int(if_generation_match)
        if (expected == 0 and existing is not None) or (
            expected not in (None, 0)
            and (existing is None or existing.generation != expected)
        ):
            raise StorageManifestConflictError("generation precondition failed")
        generation = 1 if existing is None else existing.generation + 1
        self.objects[key] = _Object(bytes(data), generation)
        return self.get_object_metadata(bucket, object_name)  # type: ignore[return-value]

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        self.operations.append(f"download:{object_name}")
        return self.objects[(bucket, object_name)].data


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "svrpo.parquet", tmp_path / "svrpo.metadata.json"


def _run(
    tmp_path: Path,
    payload: bytes,
    *,
    dry_run: bool = True,
    storage: FakeStorage | None = None,
):
    parquet, metadata = _paths(tmp_path)
    return sync_and_publish_svrpo_history(
        parquet_path=parquet,
        metadata_path=metadata,
        dry_run=dry_run,
        http_get=lambda: payload,
        storage_client=storage,
        now=NOW,
    )


def test_valid_csv_creates_cboe_metadata_and_dry_run_uses_no_storage(
    tmp_path: Path,
) -> None:
    storage = FakeStorage()

    result = _run(tmp_path, b"\xef\xbb\xbf" + CSV_V1, storage=storage)
    metadata = json.loads(_paths(tmp_path)[1].read_text(encoding="utf-8"))

    assert result.first_date == "2005-12-30"
    assert result.last_date == "2026-09-14"
    assert result.row_count == 2
    assert metadata["source"] == "Cboe"
    assert metadata["source_url"] == SVRPO_SOURCE_URL
    assert metadata["market_data_as_of"] == "2026-09-14"
    assert metadata["file_sha256"]
    assert "backtested history" in metadata["warnings"][0]
    assert storage.operations == []


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"<html>error</html>", "DATE and SVRPO"),
        (b"DATE,SVRPO\n09/14/2026,0\n", "nonfinite or nonpositive"),
        (
            b"DATE,SVRPO\n09/14/2026,1\n09/14/2026,2\n",
            "duplicate observation dates",
        ),
    ],
)
def test_invalid_downloads_are_rejected(payload: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_svrpo_csv(payload, market_date=NOW.date())


def test_failed_download_and_coverage_regression_preserve_existing_state(
    tmp_path: Path,
) -> None:
    _run(tmp_path, CSV_V1)
    parquet, metadata = _paths(tmp_path)
    before = (parquet.read_bytes(), metadata.read_bytes())

    with pytest.raises(RuntimeError, match="download failed"):
        sync_and_publish_svrpo_history(
            parquet_path=parquet,
            metadata_path=metadata,
            dry_run=True,
            http_get=lambda: (_ for _ in ()).throw(RuntimeError("download failed")),
            now=NOW,
        )
    assert (parquet.read_bytes(), metadata.read_bytes()) == before

    truncated = b"DATE,SVRPO\n09/14/2026,120.5\n"
    with pytest.raises(ValueError, match="lost previously accepted"):
        _run(tmp_path, truncated)
    assert (parquet.read_bytes(), metadata.read_bytes()) == before


def test_download_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def fail_download(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("synthetic timeout")

    monkeypatch.setattr("urllib.request.urlopen", fail_download)
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        download_svrpo_csv(max_attempts=2, retry_delay_seconds=0)
    assert attempts == 2

def test_correction_changes_content_while_unchanged_data_preserves_bytes(
    tmp_path: Path,
) -> None:
    _run(tmp_path, CSV_V1)
    parquet, metadata = _paths(tmp_path)
    first_bytes = (parquet.read_bytes(), metadata.read_bytes())

    unchanged = _run(tmp_path, CSV_V1)
    assert unchanged.local_changed is False
    assert (parquet.read_bytes(), metadata.read_bytes()) == first_bytes

    corrected = _run(tmp_path, CSV_REVISED)
    assert corrected.local_changed is True
    assert corrected.revised_count == 1
    assert corrected.last_date == unchanged.last_date
    assert parquet.read_bytes() != first_bytes[0]


def test_dry_run_then_normal_publish_and_repeat_are_idempotent(tmp_path: Path) -> None:
    storage = FakeStorage()

    dry_run = _run(tmp_path, CSV_V1, storage=storage)
    assert dry_run.gcs_verify is None
    assert storage.operations == []

    published = _run(tmp_path, CSV_V1, dry_run=False, storage=storage)
    assert published.gcs_publish.status == "published"
    assert published.gcs_verify is not None and published.gcs_verify.verified

    repeated = _run(tmp_path, CSV_V1, dry_run=False, storage=storage)
    assert repeated.local_changed is False
    assert repeated.gcs_publish.status == "already_published"
    assert repeated.gcs_verify is not None and repeated.gcs_verify.verified


def test_retry_after_publication_failure_reuses_prepared_local_store(
    tmp_path: Path,
) -> None:
    storage = FakeStorage()
    storage.fail_next_upload = True

    with pytest.raises(RuntimeError, match="synthetic upload failure"):
        _run(tmp_path, CSV_V1, dry_run=False, storage=storage)
    parquet, metadata = _paths(tmp_path)
    prepared_bytes = (parquet.read_bytes(), metadata.read_bytes())

    result = _run(tmp_path, CSV_V1, dry_run=False, storage=storage)
    assert result.local_changed is False
    assert result.gcs_verify is not None and result.gcs_verify.verified
    assert (parquet.read_bytes(), metadata.read_bytes()) == prepared_bytes
