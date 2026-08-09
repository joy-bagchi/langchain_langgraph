"""Publish one immutable option-surface snapshot per observation time."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import Any

import pandas as pd

from agentic_vol_regime_app.data.ibkr_client import IBKRConnectionConfig, IBKRDataPipe, IBKROptionChainRequest
from agentic_vol_regime_app.data.sector_history_gcs import GoogleCloudStorageClient, StorageClientProtocol, StorageManifestConflictError

from .contracts import MANIFEST_SCHEMA_VERSION, OPTION_CHAIN_SCHEMA_VERSION, option_chain_frame, validate_option_chain_frame

DEFAULT_BUCKET = "marketphysics-market-manifold-data"
DEFAULT_PREFIX = "market-manifold/option-chain-iv"


def _sha(payload: bytes) -> str: return hashlib.sha256(payload).hexdigest()
def _json(payload: dict[str, Any]) -> bytes: return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _upload_immutable(client: StorageClientProtocol, bucket: str, object_name: str, payload: bytes, content_type: str) -> None:
    current = client.get_object_metadata(bucket, object_name)
    if current is None:
        client.upload_bytes(bucket=bucket, object_name=object_name, data=payload, if_generation_match=0, content_type=content_type)
    elif _sha(client.download_bytes(bucket, object_name)) != _sha(payload):
        raise RuntimeError(f"Immutable GCS object conflict at `{object_name}`.")


def publish_option_chain(frame: pd.DataFrame, *, bucket: str = DEFAULT_BUCKET, prefix: str = DEFAULT_PREFIX,
                         project: str | None = None, dry_run: bool = False, storage_client: StorageClientProtocol | None = None) -> dict[str, Any]:
    """Publish one dated surface and add it to the discoverable catalog."""
    validate_option_chain_frame(frame)
    buf = BytesIO(); frame.to_parquet(buf, index=False); parquet = buf.getvalue(); parquet_sha = _sha(parquet)
    observation_time, observation_date = str(frame.observation_time.min()), str(frame.observation_date.min())
    dataset_id = f"option-chain-iv-{observation_time.replace(':', '').replace('+00:00', 'Z')}-{parquet_sha[:8]}"
    base = prefix.strip("/"); root = f"{base}/datasets/{dataset_id}"
    parquet_object, metadata_object, catalog_object = f"{root}/option_chain_iv.parquet", f"{root}/metadata.json", f"{base}/manifests/catalog.json"
    metadata = {"schema_version": OPTION_CHAIN_SCHEMA_VERSION, "dataset_id": dataset_id, "observation_time": observation_time,
                "observation_date": observation_date, "row_count": len(frame), "symbols": sorted(frame.symbol.unique()), "content_sha256": parquet_sha}
    entry = {"dataset_id": dataset_id, "observation_time": observation_time, "observation_date": observation_date,
             "parquet": {"bucket": bucket, "object": parquet_object, "sha256": parquet_sha}}
    result = {"status": "dry_run" if dry_run else "published", "dataset_id": dataset_id, "row_count": len(frame),
              "parquet_uri": f"gs://{bucket}/{parquet_object}", "catalog_uri": f"gs://{bucket}/{catalog_object}"}
    if dry_run: return result
    client = storage_client or GoogleCloudStorageClient(project=project)
    if not client.bucket_exists(bucket): raise RuntimeError(f"GCS bucket `{bucket}` does not exist or is not visible.")
    _upload_immutable(client, bucket, parquet_object, parquet, "application/octet-stream")
    _upload_immutable(client, bucket, metadata_object, _json(metadata), "application/json")
    for _ in range(3):
        current = client.get_object_metadata(bucket, catalog_object)
        catalog = {"manifest_schema_version": MANIFEST_SCHEMA_VERSION, "dataset_schema_version": OPTION_CHAIN_SCHEMA_VERSION, "datasets": []} if current is None else json.loads(client.download_bytes(bucket, catalog_object).decode("utf-8"))
        datasets = [item for item in catalog["datasets"] if item["dataset_id"] != dataset_id] + [entry]
        catalog["datasets"] = sorted(datasets, key=lambda item: (item["observation_time"], item["dataset_id"]))
        try:
            client.upload_bytes(bucket=bucket, object_name=catalog_object, data=_json(catalog), if_generation_match=0 if current is None else current.generation, content_type="application/json")
            break
        except StorageManifestConflictError:
            continue
    else: raise RuntimeError("Option-surface catalog changed concurrently; retry publication.")
    if _sha(client.download_bytes(bucket, parquet_object)) != parquet_sha: raise RuntimeError("Published Parquet failed SHA-256 verification.")
    return result


def collect_and_publish(*, symbol: str = "SPY", host: str = "127.0.0.1", port: int = 4001, client_id: int = 74,
                        expiry_count: int = 8, strike_count: int = 25, bucket: str = DEFAULT_BUCKET, prefix: str = DEFAULT_PREFIX,
                        project: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    pipe = IBKRDataPipe(connection=IBKRConnectionConfig(host=host, port=port, client_id=client_id))
    snapshot = pipe.fetch_market_snapshot(IBKROptionChainRequest(symbol=symbol, expiry_count=expiry_count, strike_count=strike_count))
    return publish_option_chain(option_chain_frame(snapshot.to_dict()), bucket=bucket, prefix=prefix, project=project, dry_run=dry_run)
