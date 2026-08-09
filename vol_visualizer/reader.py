"""Catalog-backed loading of historical option-surface observations."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pandas as pd

from agentic_vol_regime_app.data.sector_history_gcs import GoogleCloudStorageClient, StorageClientProtocol

from vol_surface_publisher.contracts import MANIFEST_SCHEMA_VERSION, validate_option_chain_frame
from vol_surface_publisher.publisher import DEFAULT_BUCKET, DEFAULT_PREFIX


def load_option_chain_history(*, bucket: str = DEFAULT_BUCKET, prefix: str = DEFAULT_PREFIX, project: str | None = None,
                              start_date: str | None = None, end_date: str | None = None,
                              storage_client: StorageClientProtocol | None = None) -> pd.DataFrame:
    """Load verified option-surface snapshots listed in the publisher catalog."""
    client = storage_client or GoogleCloudStorageClient(project=project)
    catalog_object = f"{prefix.strip('/')}/manifests/catalog.json"
    catalog = json.loads(client.download_bytes(bucket, catalog_object).decode("utf-8"))
    if catalog.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unexpected option-surface catalog schema version.")
    entries = list(catalog.get("datasets", []))
    if start_date:
        entries = [item for item in entries if item["observation_date"] >= start_date]
    if end_date:
        entries = [item for item in entries if item["observation_date"] <= end_date]
    if not entries:
        raise ValueError("No published option-surface datasets match the requested date range.")
    frames = []
    for entry in entries:
        descriptor = dict(entry["parquet"])
        payload = client.download_bytes(str(descriptor["bucket"]), str(descriptor["object"]))
        if hashlib.sha256(payload).hexdigest() != descriptor.get("sha256"):
            raise ValueError(f"Option-surface checksum mismatch for `{entry['dataset_id']}`.")
        frames.append(pd.read_parquet(BytesIO(payload)))
    frame = pd.concat(frames, ignore_index=True)
    validate_option_chain_frame(frame)
    return frame
