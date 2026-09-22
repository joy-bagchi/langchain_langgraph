"""Read-only loaders for the established Market Physics GCS datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from typing import Any

import pandas as pd
from google.cloud import storage


@dataclass(frozen=True)
class SourceObject:
    bucket: str
    object: str
    generation: str | None
    sha256: str

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.object}"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "uri": self.uri,
            "generation": self.generation,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SurfaceSnapshot:
    dataset_id: str
    observation_date: str
    observation_time: str
    frame: pd.DataFrame
    source_object: SourceObject


@dataclass(frozen=True)
class MarketInputs:
    surfaces: tuple[SurfaceSnapshot, ...]
    history: pd.DataFrame
    source_objects: tuple[SourceObject, ...]
    history_metadata: dict[str, Any]


class GCSMarketDataReader:
    def __init__(
        self,
        project: str,
        bucket: str,
        option_catalog_object: str,
        volatility_history_manifest_object: str,
    ) -> None:
        self.bucket_name = bucket
        self.option_catalog_object = option_catalog_object
        self.volatility_history_manifest_object = (
            volatility_history_manifest_object
        )
        self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket)

    def _download(
        self, object_name: str, expected_sha256: str | None = None
    ) -> tuple[bytes, SourceObject]:
        blob = self.bucket.blob(object_name)
        payload = blob.download_as_bytes()
        digest = sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(
                f"checksum mismatch for gs://{self.bucket_name}/{object_name}"
            )
        blob.reload()
        return payload, SourceObject(
            bucket=self.bucket_name,
            object=object_name,
            generation=str(blob.generation) if blob.generation else None,
            sha256=digest,
        )

    @staticmethod
    def _json(payload: bytes, label: str) -> dict[str, Any]:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        return value

    def load(self) -> MarketInputs:
        catalog_bytes, catalog_source = self._download(
            self.option_catalog_object
        )
        catalog = self._json(catalog_bytes, "option catalog")
        if catalog.get("manifest_schema_version") != "option_chain_iv_catalog.v1":
            raise ValueError("unsupported option catalog schema")

        snapshots: list[SurfaceSnapshot] = []
        sources: list[SourceObject] = [catalog_source]
        seen_dates: set[str] = set()
        for entry in catalog.get("datasets", []):
            observation_date = str(entry["observation_date"])
            if observation_date in seen_dates:
                raise ValueError(
                    f"catalog contains multiple selected snapshots for {observation_date}"
                )
            seen_dates.add(observation_date)
            descriptor = dict(entry["parquet"])
            if descriptor.get("bucket") != self.bucket_name:
                raise ValueError("cross-bucket option descriptor is not supported")
            payload, source = self._download(
                str(descriptor["object"]), str(descriptor["sha256"])
            )
            frame = pd.read_parquet(BytesIO(payload))
            snapshots.append(
                SurfaceSnapshot(
                    dataset_id=str(entry["dataset_id"]),
                    observation_date=observation_date,
                    observation_time=str(entry["observation_time"]),
                    frame=frame,
                    source_object=source,
                )
            )
            sources.append(source)

        manifest_bytes, manifest_source = self._download(
            self.volatility_history_manifest_object
        )
        manifest = self._json(manifest_bytes, "volatility history manifest")
        if manifest.get("manifest_schema_version") != "sector_prices_manifest.v1":
            raise ValueError("unsupported volatility history manifest schema")
        parquet_descriptor = dict(manifest["parquet"])
        parquet_bytes, parquet_source = self._download(
            str(parquet_descriptor["object"]), str(parquet_descriptor["sha256"])
        )
        metadata_descriptor = dict(manifest["metadata"])
        metadata_bytes, metadata_source = self._download(
            str(metadata_descriptor["object"]), str(metadata_descriptor["sha256"])
        )
        metadata = self._json(metadata_bytes, "volatility history metadata")
        history = pd.read_parquet(BytesIO(parquet_bytes))
        required = {"date", "SPY", "VIX"}
        if not required.issubset(history.columns):
            raise ValueError(
                f"volatility history is missing columns: {sorted(required - set(history.columns))}"
            )
        sources.extend((manifest_source, parquet_source, metadata_source))
        return MarketInputs(
            surfaces=tuple(sorted(snapshots, key=lambda item: item.observation_date)),
            history=history,
            source_objects=tuple(sources),
            history_metadata=metadata,
        )


__all__ = [
    "GCSMarketDataReader",
    "MarketInputs",
    "SourceObject",
    "SurfaceSnapshot",
]
