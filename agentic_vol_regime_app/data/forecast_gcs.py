"""Immutable GCS publishing for finalized Market Physics forecasts."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from agentic_vol_regime_app.data.sector_history_gcs import (BucketMissingError, GoogleCloudStorageClient,
    StorageAuthenticationError, StorageClientProtocol, StorageError, StorageManifestConflictError,
    StorageNetworkError, StorageObjectConflictError, StoragePermissionError)
from agentic_vol_regime_app.forecast_contract import FORECAST_MANIFEST_SCHEMA_VERSION, FORECAST_SCHEMA_VERSION, canonical_json_bytes, sha256_bytes

DEFAULT_BUCKET = "marketphysics-market-manifold-data"
DEFAULT_PREFIX = "market-manifold/forecasts"

@dataclass(slots=True)
class ForecastPublishResult:
    status: str; forecast_id: str; manifest_uri: str; forecast: dict[str, Any]; report: dict[str, Any]
    uploaded_objects: list[str] = field(default_factory=list); reused_objects: list[str] = field(default_factory=list)
    verified: bool = False; warnings: list[str] = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "forecast_id": self.forecast_id, "manifest_uri": self.manifest_uri,
                "forecast": self.forecast, "report": self.report, "uploaded_objects": self.uploaded_objects,
                "reused_objects": self.reused_objects, "verified": self.verified, "warnings": self.warnings}

def _descriptor(bucket: str, name: str, data: bytes, metadata, content_type: str) -> dict[str, Any]:
    return {"bucket": bucket, "object": name, "object_name": name, "gs_uri": f"gs://{bucket}/{name}", "sha256": sha256_bytes(data), "size_bytes": len(data), "generation": str(metadata.generation), "content_type": content_type}

class ForecastGCSPublisher:
    def __init__(self, *, storage_client: StorageClientProtocol | None = None, bucket: str | None = None,
                 prefix: str | None = None, project: str | None = None) -> None:
        self.storage_client = storage_client or GoogleCloudStorageClient(project=project or os.getenv("GOOGLE_CLOUD_PROJECT", "marketphysics"))
        self.bucket = (bucket or os.getenv("MARKET_PHYSICS_FORECAST_GCS_BUCKET") or DEFAULT_BUCKET).strip()
        self.prefix = (prefix or os.getenv("MARKET_PHYSICS_FORECAST_GCS_PREFIX") or DEFAULT_PREFIX).strip("/")

    def publish(self, *, artifact: dict[str, Any], report_markdown: str, dry_run: bool = False) -> ForecastPublishResult:
        forecast_id = str(artifact["forecast_id"]); base = f"{self.prefix}/runs/{forecast_id}"
        forecast_name, report_name, manifest_name = f"{base}/forecast.json", f"{base}/report.md", f"{self.prefix}/manifests/latest.json"
        forecast_bytes, report_bytes = canonical_json_bytes(artifact), report_markdown.encode("utf-8")
        if dry_run:
            empty = {"bucket": self.bucket, "object_name": forecast_name, "gs_uri": f"gs://{self.bucket}/{forecast_name}", "sha256": sha256_bytes(forecast_bytes), "size_bytes": len(forecast_bytes), "generation": None, "content_type": "application/json"}
            return ForecastPublishResult("dry_run", forecast_id, f"gs://{self.bucket}/{manifest_name}", empty, {**empty, "object_name": report_name, "gs_uri": f"gs://{self.bucket}/{report_name}", "sha256": sha256_bytes(report_bytes), "size_bytes": len(report_bytes), "content_type": "text/markdown"}, verified=False)
        if not self.storage_client.bucket_exists(self.bucket): raise BucketMissingError(f"GCS bucket `{self.bucket}` is unavailable.")
        old_meta = self.storage_client.get_object_metadata(self.bucket, manifest_name)
        uploaded: list[str] = []; reused: list[str] = []
        def immutable(name: str, data: bytes, ctype: str) -> dict[str, Any]:
            try:
                meta = self.storage_client.upload_bytes(bucket=self.bucket, object_name=name, data=data, if_generation_match=0, content_type=ctype); uploaded.append(f"gs://{self.bucket}/{name}")
            except StorageManifestConflictError:
                meta = self.storage_client.get_object_metadata(self.bucket, name)
                if meta is None or self.storage_client.download_bytes(self.bucket, name) != data: raise StorageObjectConflictError(f"Immutable object collision at {name}.")
                reused.append(f"gs://{self.bucket}/{name}")
            return _descriptor(self.bucket, name, data, meta, ctype)
        forecast = immutable(forecast_name, forecast_bytes, "application/json")
        report = immutable(report_name, report_bytes, "text/markdown; charset=utf-8")
        manifest = {"manifest_schema_version": FORECAST_MANIFEST_SCHEMA_VERSION, "artifact_schema_version": FORECAST_SCHEMA_VERSION,
                    "forecast_id": forecast_id, "generated_at": artifact["generated_at"], "market_data_as_of": artifact["market_data_as_of"],
                    "model": artifact["model"], "decision_eligible": artifact["decision_eligible"], "review_required": artifact["review_required"],
                    "forecast": forecast, "report": report, "publisher": {"version": "forecast_gcs.v1", "provenance": artifact["provenance"]}}
        existing = json.loads(self.storage_client.download_bytes(self.bucket, manifest_name)) if old_meta else None
        if existing and str(existing.get("market_data_as_of", "")) > str(artifact["market_data_as_of"]):
            raise StorageManifestConflictError("Refusing to replace a newer forecast manifest with an older forecast.")
        manifest_bytes = canonical_json_bytes(manifest)
        try:
            manifest_meta = self.storage_client.upload_bytes(bucket=self.bucket, object_name=manifest_name, data=manifest_bytes, if_generation_match=str(old_meta.generation) if old_meta else 0, content_type="application/json")
            uploaded.append(f"gs://{self.bucket}/{manifest_name}")
        except StorageManifestConflictError:
            winner = json.loads(self.storage_client.download_bytes(self.bucket, manifest_name))
            if winner.get("forecast_id") != forecast_id: raise
            manifest_meta = self.storage_client.get_object_metadata(self.bucket, manifest_name)
        final_manifest = self.storage_client.download_bytes(self.bucket, manifest_name)
        if final_manifest != manifest_bytes or self.storage_client.download_bytes(self.bucket, forecast_name) != forecast_bytes or self.storage_client.download_bytes(self.bucket, report_name) != report_bytes:
            raise StorageNetworkError("Final forecast publication verification failed.")
        return ForecastPublishResult("published", forecast_id, f"gs://{self.bucket}/{manifest_name}", forecast, report, uploaded, reused, True)
