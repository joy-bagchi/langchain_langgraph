"""Catalog-backed loading of historical option-surface observations."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from typing import Any, Protocol

import pandas as pd

from vol_surface_publisher.contracts import MANIFEST_SCHEMA_VERSION, validate_option_chain_frame

DEFAULT_BUCKET = "marketphysics-market-manifold-data"
DEFAULT_PREFIX = "market-manifold/option-chain-iv"


class StorageClientProtocol(Protocol):
    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        ...


class GoogleCloudStorageClient:
    """Minimal GCS reader, intentionally independent from the regime app."""

    def __init__(self, *, project: str | None = None) -> None:
        self.project = project
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.auth.exceptions import DefaultCredentialsError
                from google.cloud import storage
                self._client = storage.Client(project=self.project)
            except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Install google-cloud-storage to read IV surfaces from GCS.") from exc
            except DefaultCredentialsError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Google ADC is unavailable. Run `gcloud auth application-default login`.") from exc
        return self._client

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        return bytes(self.client.bucket(bucket).blob(object_name).download_as_bytes())


def load_surface_catalog(*, bucket: str = DEFAULT_BUCKET, prefix: str = DEFAULT_PREFIX,
                         project: str | None = None, storage_client: StorageClientProtocol | None = None) -> pd.DataFrame:
    """Load the publisher catalog without downloading any Parquet objects."""
    client = storage_client or GoogleCloudStorageClient(project=project)
    catalog_object = f"{prefix.strip('/')}/manifests/catalog.json"
    catalog = json.loads(client.download_bytes(bucket, catalog_object).decode("utf-8"))
    if catalog.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unexpected option-surface catalog schema version.")
    rows = list(catalog.get("datasets", []))
    if not rows:
        return pd.DataFrame(columns=["dataset_id", "observation_date", "observation_time", "symbols"])
    return pd.DataFrame(rows).sort_values(["observation_date", "observation_time"]).reset_index(drop=True)


def load_option_chain_history(*, bucket: str = DEFAULT_BUCKET, prefix: str = DEFAULT_PREFIX, project: str | None = None,
                              start_date: str | None = None, end_date: str | None = None,
                              storage_client: StorageClientProtocol | None = None) -> pd.DataFrame:
    """Load verified option-surface snapshots listed in the publisher catalog."""
    client = storage_client or GoogleCloudStorageClient(project=project)
    entries = load_surface_catalog(
        bucket=bucket, prefix=prefix, project=project, storage_client=client
    ).to_dict(orient="records")
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
