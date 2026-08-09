"""Immutable, verified GCS publishing for finalized Market Physics forecasts."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from agentic_vol_regime_app.data.sector_history_gcs import (
    BucketMissingError,
    GoogleCloudStorageClient,
    StorageClientProtocol,
    StorageManifestConflictError,
    StorageNetworkError,
    StorageObjectConflictError,
)
from agentic_vol_regime_app.forecast_contract import (
    FORECAST_MANIFEST_SCHEMA_VERSION,
    FORECAST_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)

DEFAULT_BUCKET = "marketphysics-market-manifold-data"
DEFAULT_PREFIX = "market-manifold/forecasts"
PUBLISHER_VERSION = "forecast_gcs.v1"


def _configured(explicit: str | None, environment: str, default: str) -> str:
    """Resolve explicit input, then environment, then a safe default."""
    if explicit is not None:
        return str(explicit).strip()
    return os.getenv(environment, default).strip()


@dataclass(frozen=True, slots=True)
class ForecastObjectDescriptor:
    bucket: str
    object_name: str
    gs_uri: str
    sha256: str
    size_bytes: int
    generation: str | None
    content_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "object_name": self.object_name,
            "object": self.object_name,
            "gs_uri": self.gs_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "generation": self.generation,
            "content_type": self.content_type,
        }


@dataclass(slots=True)
class ForecastPublishResult:
    status: str
    forecast_id: str
    bucket: str
    prefix: str
    manifest_uri: str
    forecast: ForecastObjectDescriptor
    report: ForecastObjectDescriptor
    manifest: ForecastObjectDescriptor
    uploaded_objects: list[str] = field(default_factory=list)
    reused_objects: list[str] = field(default_factory=list)
    verified: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "forecast_id": self.forecast_id,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "manifest_uri": self.manifest_uri,
            "forecast": self.forecast.to_dict(),
            "report": self.report.to_dict(),
            "manifest": self.manifest.to_dict(),
            "uploaded_objects": list(self.uploaded_objects),
            "reused_objects": list(self.reused_objects),
            "verified": self.verified,
            "warnings": list(self.warnings),
        }


class ForecastGCSPublisher:
    """Publishes one immutable forecast run and safely advances its pointer."""

    def __init__(self, *, storage_client: StorageClientProtocol | None = None,
                 bucket: str | None = None, prefix: str | None = None,
                 project: str | None = None) -> None:
        self.bucket = _configured(bucket, "MARKET_PHYSICS_FORECAST_GCS_BUCKET", DEFAULT_BUCKET)
        self.prefix = _configured(prefix, "MARKET_PHYSICS_FORECAST_GCS_PREFIX", DEFAULT_PREFIX).strip("/")
        self.storage_client = storage_client or GoogleCloudStorageClient(
            project=_configured(project, "GOOGLE_CLOUD_PROJECT", "marketphysics")
        )
        if not self.bucket or not self.prefix:
            raise ValueError("Forecast GCS bucket and prefix must be non-empty.")

    def _descriptor(self, name: str, payload: bytes, metadata: Any | None,
                    content_type: str) -> ForecastObjectDescriptor:
        return ForecastObjectDescriptor(
            bucket=self.bucket,
            object_name=name,
            gs_uri=f"gs://{self.bucket}/{name}",
            sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            generation=str(metadata.generation) if metadata is not None else None,
            content_type=content_type,
        )

    def _publish_immutable(self, *, name: str, payload: bytes, content_type: str,
                           uploaded: list[str], reused: list[str]) -> ForecastObjectDescriptor:
        try:
            metadata = self.storage_client.upload_bytes(
                bucket=self.bucket, object_name=name, data=payload,
                if_generation_match=0, content_type=content_type,
            )
            uploaded.append(f"gs://{self.bucket}/{name}")
        except StorageManifestConflictError:
            metadata = self.storage_client.get_object_metadata(self.bucket, name)
            if metadata is None or self.storage_client.download_bytes(self.bucket, name) != payload:
                raise StorageObjectConflictError(f"Immutable forecast object differs at `{name}`.")
            reused.append(f"gs://{self.bucket}/{name}")
        if metadata.size_bytes != len(payload):
            raise StorageNetworkError(f"GCS object size verification failed for `{name}`.")
        return self._descriptor(name, payload, metadata, content_type)

    def publish(self, *, artifact: dict[str, Any], report_markdown: str | bytes,
                dry_run: bool = False) -> ForecastPublishResult:
        forecast_id = str(artifact.get("forecast_id", "")).strip()
        if not forecast_id or "/" in forecast_id or "\\" in forecast_id:
            raise ValueError("forecast_id must be a non-empty path-safe identifier.")
        run_root = f"{self.prefix}/runs/{forecast_id}"
        forecast_name = f"{run_root}/forecast.json"
        report_name = f"{run_root}/report.md"
        manifest_name = f"{self.prefix}/manifests/latest.json"
        forecast_bytes = canonical_json_bytes(artifact)
        report_bytes = report_markdown if isinstance(report_markdown, bytes) else str(report_markdown).encode("utf-8")
        forecast_stub = self._descriptor(forecast_name, forecast_bytes, None, "application/json")
        report_stub = self._descriptor(report_name, report_bytes, None, "text/markdown; charset=utf-8")
        manifest_stub = self._descriptor(manifest_name, b"", None, "application/json")
        if dry_run:
            return ForecastPublishResult("dry_run", forecast_id, self.bucket, self.prefix,
                                         manifest_stub.gs_uri, forecast_stub, report_stub,
                                         manifest_stub, verified=False)
        if not self.storage_client.bucket_exists(self.bucket):
            raise BucketMissingError(f"GCS bucket `{self.bucket}` is unavailable.")

        old_manifest_metadata = self.storage_client.get_object_metadata(self.bucket, manifest_name)
        old_manifest = None
        if old_manifest_metadata is not None:
            old_manifest = json.loads(self.storage_client.download_bytes(self.bucket, manifest_name))
            if str(old_manifest.get("market_data_as_of", "")) > str(artifact["market_data_as_of"]):
                raise StorageManifestConflictError("Refusing to replace a strictly newer forecast manifest.")

        uploaded: list[str] = []
        reused: list[str] = []
        forecast = self._publish_immutable(name=forecast_name, payload=forecast_bytes,
                                            content_type="application/json", uploaded=uploaded, reused=reused)
        report = self._publish_immutable(name=report_name, payload=report_bytes,
                                          content_type="text/markdown; charset=utf-8", uploaded=uploaded, reused=reused)
        manifest_payload = {
            "manifest_schema_version": FORECAST_MANIFEST_SCHEMA_VERSION,
            "artifact_schema_version": FORECAST_SCHEMA_VERSION,
            "forecast_id": forecast_id,
            "generated_at": artifact["generated_at"],
            "market_data_as_of": artifact["market_data_as_of"],
            "model": artifact["model"],
            "decision_eligible": artifact["decision_eligible"],
            "review_required": artifact["review_required"],
            "forecast": forecast.to_dict(),
            "report": report.to_dict(),
            "publisher": {"version": PUBLISHER_VERSION, "provenance": artifact["provenance"]},
        }
        manifest_bytes = canonical_json_bytes(manifest_payload)
        try:
            manifest_metadata = self.storage_client.upload_bytes(
                bucket=self.bucket, object_name=manifest_name, data=manifest_bytes,
                if_generation_match=(str(old_manifest_metadata.generation) if old_manifest_metadata else 0),
                content_type="application/json",
            )
            uploaded.append(f"gs://{self.bucket}/{manifest_name}")
        except StorageManifestConflictError:
            # A single re-read turns an identical concurrent retry into success.
            winner_bytes = self.storage_client.download_bytes(self.bucket, manifest_name)
            winner = json.loads(winner_bytes)
            if winner.get("forecast_id") != forecast_id:
                raise StorageManifestConflictError("Forecast manifest changed concurrently; scheduler should retry.")
            manifest_metadata = self.storage_client.get_object_metadata(self.bucket, manifest_name)
            if manifest_metadata is None:
                raise StorageManifestConflictError("Forecast manifest disappeared during race recovery.")
            reused.append(f"gs://{self.bucket}/{manifest_name}")

        manifest = self._descriptor(manifest_name, manifest_bytes, manifest_metadata, "application/json")
        final_manifest = self.storage_client.download_bytes(self.bucket, manifest_name)
        if final_manifest != manifest_bytes:
            raise StorageNetworkError("Final forecast manifest verification failed.")
        if self.storage_client.download_bytes(self.bucket, forecast_name) != forecast_bytes:
            raise StorageNetworkError("Final forecast object verification failed.")
        if self.storage_client.download_bytes(self.bucket, report_name) != report_bytes:
            raise StorageNetworkError("Final report object verification failed.")
        return ForecastPublishResult("published", forecast_id, self.bucket, self.prefix,
                                     manifest.gs_uri, forecast, report, manifest, uploaded,
                                     reused, verified=True)
