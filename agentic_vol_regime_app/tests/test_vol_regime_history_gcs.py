from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDailyBar,
    IBKRDailyHistoryRequest,
    IBKRDailyHistoryResult,
    IBKRLiveClient,
)
from agentic_vol_regime_app.data.sector_history_gcs import StorageManifestConflictError, StorageObjectMetadata
from agentic_vol_regime_app.data.vol_regime_history_gcs import (
    DEFAULT_VOL_REGIME_GCS_BUCKET,
    DEFAULT_VOL_REGIME_GCS_PREFIX,
    VOL_REGIME_HISTORY_SYMBOLS,
    sync_and_publish_vol_regime_history,
)


class FakeDailyBars:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def request_daily_bars(self, request: IBKRDailyHistoryRequest) -> IBKRDailyHistoryResult:
        self.calls.append(request.symbol)
        dates = pd.bdate_range("2026-07-06", periods=5).date
        base = {"SPY": 500.0, "VIX": 18.0, "VVIX": 90.0}[request.symbol]
        return IBKRDailyHistoryResult(
            bars=tuple(
                IBKRDailyBar(request.symbol, day, base + index, "TRADES")
                for index, day in enumerate(dates)
            ),
            actual_what_to_show="TRADES",
        )


@dataclass
class _Object:
    data: bytes
    generation: int


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], _Object] = {}
        self.uploads: list[str] = []

    def bucket_exists(self, bucket: str) -> bool:
        return True

    def get_object_metadata(self, bucket: str, object_name: str) -> StorageObjectMetadata | None:
        value = self.objects.get((bucket, object_name))
        if value is None:
            return None
        return StorageObjectMetadata(bucket, object_name, str(value.generation), len(value.data))

    def upload_bytes(self, *, bucket: str, object_name: str, data: bytes, if_generation_match: int | str | None, content_type: str) -> StorageObjectMetadata:
        key = (bucket, object_name)
        existing = self.objects.get(key)
        expected = None if if_generation_match is None else int(if_generation_match)
        if (expected == 0 and existing is not None) or (expected not in (None, 0) and (existing is None or existing.generation != expected)):
            raise StorageManifestConflictError("generation precondition failed")
        generation = 1 if existing is None else existing.generation + 1
        self.objects[key] = _Object(bytes(data), generation)
        self.uploads.append(object_name)
        return self.get_object_metadata(bucket, object_name)  # type: ignore[return-value]

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        return self.objects[(bucket, object_name)].data


def test_sync_and_publish_vol_regime_history_writes_dedicated_gcs_dataset(tmp_path: Path) -> None:
    storage = FakeStorage()
    pipe = FakeDailyBars()

    result = sync_and_publish_vol_regime_history(
        parquet_path=tmp_path / "vol.parquet",
        metadata_path=tmp_path / "vol.metadata.json",
        target_end_date="2026-07-10",
        history_days=5,
        data_pipe=pipe,
        storage_client=storage,
    )

    assert pipe.calls == list(VOL_REGIME_HISTORY_SYMBOLS)
    assert result.gcs_publish.bucket == DEFAULT_VOL_REGIME_GCS_BUCKET
    assert result.gcs_publish.prefix == DEFAULT_VOL_REGIME_GCS_PREFIX
    assert result.gcs_publish.symbols == list(VOL_REGIME_HISTORY_SYMBOLS)
    assert result.gcs_verify is not None and result.gcs_verify.verified is True
    assert any(name.endswith("/manifests/latest.json") for name in storage.uploads)


def test_sync_and_publish_vol_regime_history_dry_run_skips_gcs_writes(tmp_path: Path) -> None:
    storage = FakeStorage()

    result = sync_and_publish_vol_regime_history(
        parquet_path=tmp_path / "vol.parquet",
        metadata_path=tmp_path / "vol.metadata.json",
        target_end_date="2026-07-10",
        history_days=5,
        dry_run=True,
        data_pipe=FakeDailyBars(),
        storage_client=storage,
    )

    assert result.gcs_publish.status == "dry_run"
    assert result.gcs_verify is None
    assert storage.uploads == []


def test_daily_index_history_uses_index_contract(monkeypatch) -> None:
    client = IBKRLiveClient(IBKRConnectionConfig())
    calls: list[tuple[str, str]] = []

    class FakeIB:
        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(client, "_connect", lambda: FakeIB())
    monkeypatch.setattr(
        IBKRLiveClient,
        "_qualify_index_contract",
        staticmethod(lambda _ib, *, symbol, exchange, currency: calls.append((symbol, exchange)) or object()),
    )
    monkeypatch.setattr(
        client,
        "_request_daily_history_bars",
        lambda *_args, **_kwargs: ([IBKRDailyBar("VIX", date(2026, 7, 10), 18.0, "TRADES")], "TRADES", []),
    )

    result = client.request_daily_bars(IBKRDailyHistoryRequest(symbol="VIX"))

    assert calls == [("VIX", "CBOE")]
    assert result.bars[0].close == 18.0
