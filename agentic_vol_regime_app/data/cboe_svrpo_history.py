from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.data.sector_history_gcs import (
    GCSPublishResult,
    StorageClientProtocol,
    publish_sector_store_to_gcs,
    verify_sector_store_in_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import (
    SECTOR_PRICE_SCHEMA_VERSION,
    SectorPriceStore,
    _frame_content_sha256,
)


SVRPO_SYMBOL = "SVRPO"
SVRPO_SOURCE_URL = (
    "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/"
    "SVRPO_History.csv"
)
SVRPO_INGESTION_VERSION = "cboe_svrpo_csv.v1"
SVRPO_DATASET_ID_PREFIX = "SVRPO-history"
SVRPO_PARQUET_FILENAME = "svrpo_history_daily.parquet"
SVRPO_METADATA_FILENAME = "metadata.json"
DEFAULT_SVRPO_GCS_PREFIX = "market-manifold/svrpo-history"
DEFAULT_SVRPO_GCS_BUCKET = "marketphysics-market-manifold-data"
SVRPO_INDEX_LAUNCH_DATE = "2018-05-29"


def default_svrpo_history_paths(
    app_paths: AppPaths | None = None,
) -> tuple[Path, Path]:
    paths = app_paths or AppPaths.default()
    base_dir = paths.root / "data" / "market_history"
    return (
        base_dir / SVRPO_PARQUET_FILENAME,
        base_dir / "svrpo_history_daily.metadata.json",
    )


@dataclass(slots=True)
class SVRPOIngestionResult:
    status: str
    source_url: str
    first_date: str
    last_date: str
    row_count: int
    inserted_count: int
    revised_count: int
    local_changed: bool
    parquet_path: str
    metadata_path: str
    gcs_publish: GCSPublishResult
    gcs_verify: GCSPublishResult | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": "Cboe",
            "source_url": self.source_url,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "market_data_as_of": self.last_date,
            "row_count": self.row_count,
            "inserted_count": self.inserted_count,
            "revised_count": self.revised_count,
            "local_changed": self.local_changed,
            "parquet_path": self.parquet_path,
            "metadata_path": self.metadata_path,
            "gcs_publish": self.gcs_publish.to_dict(),
            "gcs_verify": self.gcs_verify.to_dict() if self.gcs_verify else None,
            "warnings": list(self.warnings),
        }


def download_svrpo_csv(
    *,
    url: str = SVRPO_SOURCE_URL,
    timeout_seconds: float = 20.0,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.5,
) -> bytes:
    attempts = max(int(max_attempts), 1)
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/csv", "User-Agent": "market-manifold-svrpo/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                status = int(getattr(response, "status", 200))
                if status != 200:
                    raise urllib.error.HTTPError(
                        url, status, f"unexpected HTTP status {status}", response.headers, None
                    )
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == attempts:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt == attempts:
                break
        time.sleep(max(float(retry_delay_seconds), 0.0) * attempt)
    raise RuntimeError(
        f"Failed to download the Cboe SVRPO CSV after {attempts} attempt(s): {last_error}"
    ) from last_error


def parse_svrpo_csv(payload: bytes, *, market_date: date | None = None) -> pd.DataFrame:
    if not payload:
        raise ValueError("Cboe SVRPO response was empty.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Cboe SVRPO response was not valid UTF-8 CSV.") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = [str(item).strip() for item in (reader.fieldnames or [])]
    if not {"DATE", SVRPO_SYMBOL}.issubset(fieldnames):
        raise ValueError("Cboe SVRPO CSV must contain DATE and SVRPO columns.")

    rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        raw_date = str(row.get("DATE", "")).strip()
        raw_value = str(row.get(SVRPO_SYMBOL, "")).strip()
        try:
            observation_date = datetime.strptime(raw_date, "%m/%d/%Y").date()
        except ValueError as exc:
            raise ValueError(
                f"Cboe SVRPO CSV has an invalid DATE at row {row_number}: {raw_date!r}."
            ) from exc
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Cboe SVRPO CSV has an invalid SVRPO value at row {row_number}: {raw_value!r}."
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Cboe SVRPO CSV has a nonfinite or nonpositive value at row {row_number}."
            )
        rows.append({"date": pd.Timestamp(observation_date), SVRPO_SYMBOL: value})

    if not rows:
        raise ValueError("Cboe SVRPO CSV contains no observations.")
    frame = pd.DataFrame(rows, columns=["date", SVRPO_SYMBOL])
    if frame["date"].duplicated().any():
        raise ValueError("Cboe SVRPO CSV contains duplicate observation dates.")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    resolved_market_date = market_date or datetime.now(
        ZoneInfo("America/New_York")
    ).date()
    last_date = frame["date"].max().date()
    if last_date > resolved_market_date:
        raise ValueError(
            "Cboe SVRPO CSV contains a future observation date: "
            f"{last_date.isoformat()} > {resolved_market_date.isoformat()}."
        )
    return frame


def _compare_with_existing(
    store: SectorPriceStore, frame: pd.DataFrame
) -> tuple[bool, int, int]:
    parquet_exists = store.parquet_path.exists()
    metadata_exists = store.metadata_path.exists()
    if parquet_exists != metadata_exists:
        raise RuntimeError(
            "SVRPO local store is incomplete. Repair or remove both isolated SVRPO "
            f"files before rerunning: parquet={parquet_exists} metadata={metadata_exists}."
        )
    if not store.exists():
        return True, len(frame), 0

    existing = store.load_offline()
    old_dates = set(existing["date"].dt.date)
    new_dates = set(frame["date"].dt.date)
    missing_dates = sorted(old_dates - new_dates)
    if missing_dates:
        raise ValueError(
            "Cboe SVRPO refresh lost previously accepted observation dates; "
            f"missing_count={len(missing_dates)} first_missing={missing_dates[0].isoformat()}."
        )
    old_last = existing["date"].max().date()
    new_last = frame["date"].max().date()
    if new_last < old_last:
        raise ValueError(
            f"Cboe SVRPO refresh regressed last date from {old_last} to {new_last}."
        )
    if len(frame) < len(existing):
        raise ValueError(
            f"Cboe SVRPO refresh truncated coverage from {len(existing)} to {len(frame)} rows."
        )

    comparison = existing.merge(frame, on="date", how="inner", suffixes=("_old", "_new"))
    revised_count = int(
        (~comparison[f"{SVRPO_SYMBOL}_old"].eq(comparison[f"{SVRPO_SYMBOL}_new"])).sum()
    )
    inserted_count = len(new_dates - old_dates)
    return bool(revised_count or inserted_count), inserted_count, revised_count


def _build_metadata(
    *, frame: pd.DataFrame, retrieved_at: datetime, source_payload: bytes
) -> dict[str, Any]:
    first_date = frame["date"].min().date().isoformat()
    last_date = frame["date"].max().date().isoformat()
    return {
        "schema_version": SECTOR_PRICE_SCHEMA_VERSION,
        "generated_at": retrieved_at.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source": "Cboe",
        "source_url": SVRPO_SOURCE_URL,
        "ingestion_version": SVRPO_INGESTION_VERSION,
        "mode": "full_download_snapshot",
        "symbols": [SVRPO_SYMBOL],
        "first_date": first_date,
        "last_date": last_date,
        "market_data_as_of": last_date,
        "row_count": int(len(frame)),
        "content_sha256": _frame_content_sha256(frame),
        "source_csv_sha256": hashlib.sha256(source_payload).hexdigest(),
        "per_symbol": {
            SVRPO_SYMBOL: {
                "first_valid_date": first_date,
                "last_valid_date": last_date,
                "non_null_count": int(len(frame)),
                "internal_gap_count": 0,
                "source": "Cboe",
            }
        },
        "warnings": [
            "SVRPO values before the 2018-05-29 index launch are backtested history; "
            "source coverage does not establish live publication during that period."
        ],
    }


def _replace_local_store(
    *, store: SectorPriceStore, frame: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    store.parquet_path.parent.mkdir(parents=True, exist_ok=True)
    store.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="svrpo-stage-", dir=store.parquet_path.parent
    ) as stage_dir:
        staged = SectorPriceStore(
            parquet_path=Path(stage_dir) / store.parquet_path.name,
            metadata_path=Path(stage_dir) / store.metadata_path.name,
            symbols=[SVRPO_SYMBOL],
        )
        staged.write_authoritative(frame=frame, metadata=metadata)
        staged.load_offline()
        os.replace(staged.parquet_path, store.parquet_path)
        os.replace(staged.metadata_path, store.metadata_path)


def sync_and_publish_svrpo_history(
    *,
    bucket: str = DEFAULT_SVRPO_GCS_BUCKET,
    prefix: str = DEFAULT_SVRPO_GCS_PREFIX,
    project: str | None = None,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    dry_run: bool = False,
    timeout_seconds: float = 20.0,
    max_attempts: int = 3,
    http_get: Callable[[], bytes] | None = None,
    storage_client: StorageClientProtocol | None = None,
    now: datetime | None = None,
) -> SVRPOIngestionResult:
    default_parquet, default_metadata = default_svrpo_history_paths()
    store = SectorPriceStore(
        parquet_path=parquet_path or default_parquet,
        metadata_path=metadata_path or default_metadata,
        symbols=[SVRPO_SYMBOL],
    )
    retrieved_at = now or datetime.now(timezone.utc)
    payload = (
        http_get()
        if http_get is not None
        else download_svrpo_csv(
            timeout_seconds=timeout_seconds, max_attempts=max_attempts
        )
    )
    market_date = retrieved_at.astimezone(ZoneInfo("America/New_York")).date()
    frame = parse_svrpo_csv(payload, market_date=market_date)
    store.validate_frame(frame)
    changed, inserted_count, revised_count = _compare_with_existing(store, frame)
    if changed:
        metadata = _build_metadata(
            frame=frame, retrieved_at=retrieved_at, source_payload=payload
        )
        _replace_local_store(store=store, frame=frame, metadata=metadata)

    publish_result = publish_sector_store_to_gcs(
        bucket=bucket,
        prefix=prefix,
        project=project,
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=[SVRPO_SYMBOL],
        dataset_id_prefix=SVRPO_DATASET_ID_PREFIX,
        parquet_filename=SVRPO_PARQUET_FILENAME,
        metadata_filename=SVRPO_METADATA_FILENAME,
        dry_run=dry_run,
        storage_client=storage_client,
    )
    verify_result = None
    if not dry_run:
        verify_result = verify_sector_store_in_gcs(
            bucket=bucket,
            prefix=prefix,
            project=project,
            storage_client=storage_client,
        )
    validation = store.validate_frame(frame)
    return SVRPOIngestionResult(
        status="dry_run" if dry_run else publish_result.status,
        source_url=SVRPO_SOURCE_URL,
        first_date=validation.first_date,
        last_date=validation.last_date,
        row_count=validation.row_count,
        inserted_count=inserted_count,
        revised_count=revised_count,
        local_changed=changed,
        parquet_path=str(store.parquet_path),
        metadata_path=str(store.metadata_path),
        gcs_publish=publish_result,
        gcs_verify=verify_result,
        warnings=list(store.load_metadata().get("warnings", [])),
    )


def main() -> None:
    default_parquet, default_metadata = default_svrpo_history_paths()
    parser = argparse.ArgumentParser(
        description="Download, validate, and publish the Cboe SVRPO daily history."
    )
    parser.add_argument("--project", default=os.getenv("MARKET_MANIFOLD_GCP_PROJECT", "marketphysics"))
    parser.add_argument("--bucket", default=os.getenv("MARKET_MANIFOLD_GCS_BUCKET", DEFAULT_SVRPO_GCS_BUCKET))
    parser.add_argument("--prefix", default=os.getenv("MARKET_MANIFOLD_SVRPO_GCS_PREFIX", DEFAULT_SVRPO_GCS_PREFIX))
    parser.add_argument("--output", default=str(default_parquet))
    parser.add_argument("--metadata-output", default=str(default_metadata))
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = sync_and_publish_svrpo_history(
        bucket=args.bucket,
        prefix=args.prefix,
        project=args.project,
        parquet_path=args.output,
        metadata_path=args.metadata_output,
        dry_run=bool(args.dry_run),
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
