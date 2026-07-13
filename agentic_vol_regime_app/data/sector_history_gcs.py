from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from agentic_vol_regime_app.data.sector_history_store import (
    DEFAULT_SECTOR_PRICE_SYMBOLS,
    SECTOR_PRICE_SCHEMA_VERSION,
    SectorPriceStore,
    SectorPriceStoreValidation,
    _frame_content_sha256,
    _parquet_file_sha256,
)


SECTOR_PRICE_MANIFEST_SCHEMA_VERSION = "sector_prices_manifest.v1"
DEFAULT_GCS_PREFIX = "market-manifold"


class StorageError(RuntimeError):
    """Base storage failure."""


class BucketMissingError(StorageError):
    """Bucket does not exist."""


class StoragePermissionError(StorageError):
    """Permission denied for storage operation."""


class StorageAuthenticationError(StorageError):
    """Authentication or ADC failure."""


class StorageNetworkError(StorageError):
    """Transient or network failure."""


class StorageObjectConflictError(StorageError):
    """Immutable object collision."""


class StorageManifestConflictError(StorageError):
    """Manifest generation precondition conflict."""


class StorageNotFoundError(StorageError):
    """Object does not exist."""


@dataclass(frozen=True, slots=True)
class StorageObjectMetadata:
    bucket: str
    object_name: str
    generation: str
    size_bytes: int

    @property
    def gs_uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"


class StorageClientProtocol(Protocol):
    def bucket_exists(self, bucket: str) -> bool:
        ...

    def get_object_metadata(self, bucket: str, object_name: str) -> StorageObjectMetadata | None:
        ...

    def upload_bytes(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        if_generation_match: int | str | None,
        content_type: str,
    ) -> StorageObjectMetadata:
        ...

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        ...


@dataclass(slots=True)
class GCSObjectDescriptor:
    bucket: str
    object_name: str
    sha256: str
    size_bytes: int
    generation: str

    @property
    def gs_uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "object": self.object_name,
            "gs_uri": self.gs_uri,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "generation": self.generation,
        }


@dataclass(slots=True)
class GCSPublishResult:
    status: str
    dataset_id: str
    market_data_as_of: str
    bucket: str
    prefix: str
    parquet_uri: str
    metadata_uri: str
    manifest_uri: str
    parquet_sha256: str
    row_count: int
    symbols: list[str]
    first_date: str
    last_date: str
    uploaded_objects: list[str] = field(default_factory=list)
    reused_objects: list[str] = field(default_factory=list)
    verified: bool = False
    ibkr_request_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dataset_id": self.dataset_id,
            "market_data_as_of": self.market_data_as_of,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "parquet_uri": self.parquet_uri,
            "metadata_uri": self.metadata_uri,
            "manifest_uri": self.manifest_uri,
            "parquet_sha256": self.parquet_sha256,
            "row_count": int(self.row_count),
            "symbols": list(self.symbols),
            "first_date": self.first_date,
            "last_date": self.last_date,
            "uploaded_objects": list(self.uploaded_objects),
            "reused_objects": list(self.reused_objects),
            "verified": bool(self.verified),
            "ibkr_request_count": int(self.ibkr_request_count),
            "warnings": list(self.warnings),
        }


def _json_default(value: Any) -> Any:
    return str(value)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dataset_id(*, market_data_as_of: str, parquet_sha256: str) -> str:
    return f"sector-prices-{market_data_as_of}-{parquet_sha256[:8]}"


def _dataset_paths(*, prefix: str, dataset_id: str) -> tuple[str, str, str]:
    normalized_prefix = str(prefix).strip("/").replace("\\", "/")
    dataset_root = f"{normalized_prefix}/datasets/{dataset_id}"
    return (
        f"{dataset_root}/sector_prices_daily.parquet",
        f"{dataset_root}/metadata.json",
        f"{normalized_prefix}/manifests/latest.json",
    )


def _validate_local_metadata(
    *,
    store: SectorPriceStore,
    validation: SectorPriceStoreValidation,
    metadata_payload: dict[str, Any],
    parquet_file_sha256: str | None = None,
) -> None:
    if str(metadata_payload.get("schema_version", "")).strip() != SECTOR_PRICE_SCHEMA_VERSION:
        raise ValueError("Local sector history metadata schema_version does not match the dataset contract.")
    if int(metadata_payload.get("row_count", -1) or -1) != int(validation.row_count):
        raise ValueError("Local sector history metadata row_count does not match the parquet store.")
    if str(metadata_payload.get("first_date", "")).strip() != validation.first_date:
        raise ValueError("Local sector history metadata first_date does not match the parquet store.")
    if str(metadata_payload.get("last_date", "")).strip() != validation.last_date:
        raise ValueError("Local sector history metadata last_date does not match the parquet store.")
    metadata_symbols = [str(item).upper() for item in list(metadata_payload.get("symbols", []))]
    if metadata_symbols != list(store.symbols):
        raise ValueError("Local sector history metadata symbols do not match the store symbol universe.")
    content_sha = str(metadata_payload.get("content_sha256", "")).strip()
    if content_sha != _frame_content_sha256(validation.frame):
        raise ValueError("Local sector history metadata content_sha256 does not match the parquet content.")
    file_sha = str(metadata_payload.get("file_sha256", "")).strip()
    actual_file_sha = parquet_file_sha256 or _parquet_file_sha256(store.parquet_path)
    if file_sha and file_sha != actual_file_sha:
        raise ValueError("Local sector history metadata file_sha256 does not match the parquet file bytes.")


def _validate_manifest_payload(payload: dict[str, Any]) -> None:
    if str(payload.get("manifest_schema_version", "")).strip() != SECTOR_PRICE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("GCS latest manifest has an unexpected manifest_schema_version.")
    if str(payload.get("dataset_schema_version", "")).strip() != SECTOR_PRICE_SCHEMA_VERSION:
        raise ValueError("GCS latest manifest has an unexpected dataset_schema_version.")
    if not str(payload.get("dataset_id", "")).strip():
        raise ValueError("GCS latest manifest is missing dataset_id.")
    for key in ("parquet", "metadata"):
        descriptor = dict(payload.get(key, {}))
        if not descriptor.get("bucket") or not descriptor.get("object") or not descriptor.get("gs_uri") or not descriptor.get("sha256"):
            raise ValueError(f"GCS latest manifest is missing required {key} descriptor fields.")


class GoogleCloudStorageClient:
    def __init__(self, *, project: str | None = None, client: Any | None = None) -> None:
        self.project = project
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage
                from google.auth.exceptions import DefaultCredentialsError
            except ModuleNotFoundError as exc:  # pragma: no cover - dependency installation path
                raise StorageAuthenticationError(
                    "google-cloud-storage is not installed. Install the app with the GCS extra or `pip install google-cloud-storage`."
                ) from exc
            try:
                self._client = storage.Client(project=self.project)
            except DefaultCredentialsError as exc:  # pragma: no cover - depends on local ADC
                raise StorageAuthenticationError(
                    "Google Application Default Credentials are not available. Run `gcloud auth application-default login`."
                ) from exc
        return self._client

    @staticmethod
    def _map_exception(exc: Exception) -> StorageError:
        try:
            from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed, Unauthorized
            from google.auth.exceptions import DefaultCredentialsError
        except ModuleNotFoundError:  # pragma: no cover
            return StorageError(str(exc))
        if isinstance(exc, DefaultCredentialsError):
            return StorageAuthenticationError(
                "Google Application Default Credentials are not available. Run `gcloud auth application-default login`."
            )
        if isinstance(exc, (Forbidden, Unauthorized)):
            return StoragePermissionError("Google Cloud Storage permission denied for the requested operation.")
        if isinstance(exc, NotFound):
            return StorageNotFoundError("The requested Google Cloud Storage object or bucket was not found.")
        if isinstance(exc, PreconditionFailed):
            return StorageManifestConflictError("Google Cloud Storage generation precondition failed.")
        return StorageNetworkError(f"Google Cloud Storage request failed: {exc}")

    def bucket_exists(self, bucket: str) -> bool:
        try:
            return bool(self.client.bucket(bucket).exists())
        except Exception as exc:  # pragma: no cover - depends on external runtime
            mapped = self._map_exception(exc)
            if isinstance(mapped, StorageNotFoundError):
                return False
            raise mapped

    def get_object_metadata(self, bucket: str, object_name: str) -> StorageObjectMetadata | None:
        try:
            blob = self.client.bucket(bucket).blob(object_name)
            if not blob.exists():
                return None
            blob.reload()
            return StorageObjectMetadata(
                bucket=bucket,
                object_name=object_name,
                generation=str(blob.generation),
                size_bytes=int(blob.size or 0),
            )
        except Exception as exc:  # pragma: no cover - depends on external runtime
            mapped = self._map_exception(exc)
            if isinstance(mapped, StorageNotFoundError):
                return None
            raise mapped

    def upload_bytes(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        if_generation_match: int | str | None,
        content_type: str,
    ) -> StorageObjectMetadata:
        try:
            blob = self.client.bucket(bucket).blob(object_name)
            kwargs: dict[str, Any] = {"content_type": content_type}
            if if_generation_match is not None:
                kwargs["if_generation_match"] = int(if_generation_match)
            blob.upload_from_string(data, **kwargs)
            blob.reload()
            return StorageObjectMetadata(
                bucket=bucket,
                object_name=object_name,
                generation=str(blob.generation),
                size_bytes=int(blob.size or 0),
            )
        except Exception as exc:  # pragma: no cover - depends on external runtime
            mapped = self._map_exception(exc)
            if isinstance(mapped, StorageManifestConflictError):
                raise mapped
            raise mapped

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        try:
            blob = self.client.bucket(bucket).blob(object_name)
            return bytes(blob.download_as_bytes())
        except Exception as exc:  # pragma: no cover - depends on external runtime
            raise self._map_exception(exc)


@dataclass(slots=True)
class _LocalDatasetContext:
    store: SectorPriceStore
    validation: SectorPriceStoreValidation
    metadata_payload: dict[str, Any]
    parquet_bytes: bytes
    metadata_bytes: bytes
    parquet_sha256: str
    metadata_sha256: str
    dataset_id: str


class SectorHistoryGCSPublisher:
    def __init__(
        self,
        *,
        store: SectorPriceStore,
        storage_client: StorageClientProtocol,
        bucket: str,
        prefix: str = DEFAULT_GCS_PREFIX,
    ) -> None:
        self.store = store
        self.storage_client = storage_client
        self.bucket = str(bucket).strip()
        self.prefix = str(prefix).strip("/") or DEFAULT_GCS_PREFIX

    def publish(self, *, dry_run: bool = False) -> GCSPublishResult:
        if not self.bucket:
            raise ValueError("GCS bucket name is required for publication.")
        local = self._load_local_dataset()
        parquet_object, metadata_object, manifest_object = _dataset_paths(prefix=self.prefix, dataset_id=local.dataset_id)
        manifest_uri = f"gs://{self.bucket}/{manifest_object}"
        if dry_run:
            return self._build_result(
                status="dry_run",
                local=local,
                parquet_uri=f"gs://{self.bucket}/{parquet_object}",
                metadata_uri=f"gs://{self.bucket}/{metadata_object}",
                manifest_uri=manifest_uri,
                warnings=[],
            )

        if not self.storage_client.bucket_exists(self.bucket):
            raise BucketMissingError(
                f"GCS bucket `{self.bucket}` does not exist or is not visible. Create it explicitly before publishing."
            )

        existing_manifest = self._read_existing_manifest(manifest_object)
        uploaded_objects: list[str] = []
        reused_objects: list[str] = []

        parquet_descriptor, parquet_created = self._publish_immutable_object(
            object_name=parquet_object,
            payload=local.parquet_bytes,
            sha256=local.parquet_sha256,
            content_type="application/octet-stream",
        )
        (uploaded_objects if parquet_created else reused_objects).append(parquet_descriptor.gs_uri)

        metadata_descriptor, metadata_created = self._publish_immutable_object(
            object_name=metadata_object,
            payload=local.metadata_bytes,
            sha256=local.metadata_sha256,
            content_type="application/json",
        )
        (uploaded_objects if metadata_created else reused_objects).append(metadata_descriptor.gs_uri)

        self._verify_remote_bytes(object_name=parquet_object, expected_sha256=local.parquet_sha256)
        self._verify_remote_bytes(object_name=metadata_object, expected_sha256=local.metadata_sha256)

        manifest_payload = {
            "manifest_schema_version": SECTOR_PRICE_MANIFEST_SCHEMA_VERSION,
            "dataset_schema_version": SECTOR_PRICE_SCHEMA_VERSION,
            "dataset_id": local.dataset_id,
            "published_at": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
            "market_data_as_of": local.validation.last_date,
            "parquet": parquet_descriptor.to_dict(),
            "metadata": metadata_descriptor.to_dict(),
        }
        manifest_bytes = _json_bytes(manifest_payload)

        if existing_manifest is not None and self._manifest_matches_local(
            manifest_payload=existing_manifest["payload"],
            dataset_id=local.dataset_id,
            parquet_sha256=local.parquet_sha256,
            metadata_sha256=local.metadata_sha256,
        ):
            return self._build_result(
                status="already_published",
                local=local,
                parquet_uri=parquet_descriptor.gs_uri,
                metadata_uri=metadata_descriptor.gs_uri,
                manifest_uri=manifest_uri,
                uploaded_objects=uploaded_objects,
                reused_objects=reused_objects,
                warnings=[],
            )

        if_generation_match = 0 if existing_manifest is None else int(existing_manifest["metadata"].generation)
        try:
            self.storage_client.upload_bytes(
                bucket=self.bucket,
                object_name=manifest_object,
                data=manifest_bytes,
                if_generation_match=if_generation_match,
                content_type="application/json",
            )
        except StorageManifestConflictError as exc:
            raise StorageManifestConflictError(
                "GCS latest manifest changed concurrently; retry after re-reading manifests/latest.json."
            ) from exc

        reloaded_manifest_bytes = self.storage_client.download_bytes(self.bucket, manifest_object)
        reloaded_manifest = json.loads(reloaded_manifest_bytes.decode("utf-8"))
        _validate_manifest_payload(reloaded_manifest)
        if reloaded_manifest != manifest_payload:
            raise ValueError("GCS latest manifest did not round-trip to the expected content after upload.")

        return self._build_result(
            status="published",
            local=local,
            parquet_uri=parquet_descriptor.gs_uri,
            metadata_uri=metadata_descriptor.gs_uri,
            manifest_uri=manifest_uri,
            uploaded_objects=uploaded_objects,
            reused_objects=reused_objects,
            warnings=[],
        )

    def verify(self) -> GCSPublishResult:
        if not self.bucket:
            raise ValueError("GCS bucket name is required for verification.")
        if not self.storage_client.bucket_exists(self.bucket):
            raise BucketMissingError(
                f"GCS bucket `{self.bucket}` does not exist or is not visible. Create it explicitly before verifying."
            )
        _parquet_object, _metadata_object, manifest_object = _dataset_paths(prefix=self.prefix, dataset_id="placeholder")
        manifest_object = f"{self.prefix}/manifests/latest.json"
        manifest_bytes = self.storage_client.download_bytes(self.bucket, manifest_object)
        manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
        _validate_manifest_payload(manifest_payload)

        parquet_descriptor = dict(manifest_payload["parquet"])
        metadata_descriptor = dict(manifest_payload["metadata"])
        parquet_bytes = self.storage_client.download_bytes(self.bucket, str(parquet_descriptor["object"]))
        metadata_bytes = self.storage_client.download_bytes(self.bucket, str(metadata_descriptor["object"]))
        if _sha256_bytes(parquet_bytes) != str(parquet_descriptor["sha256"]):
            raise ValueError("Remote parquet checksum does not match the GCS latest manifest.")
        if _sha256_bytes(metadata_bytes) != str(metadata_descriptor["sha256"]):
            raise ValueError("Remote metadata checksum does not match the GCS latest manifest.")

        metadata_payload = json.loads(metadata_bytes.decode("utf-8"))
        symbols = tuple(str(item).upper() for item in list(metadata_payload.get("symbols", DEFAULT_SECTOR_PRICE_SYMBOLS)))
        frame = pd.read_parquet(BytesIO(parquet_bytes))
        validation = SectorPriceStore(symbols=symbols).validate_frame(frame)
        _validate_local_metadata(
            store=SectorPriceStore(symbols=symbols),
            validation=validation,
            metadata_payload=metadata_payload,
            parquet_file_sha256=_sha256_bytes(parquet_bytes),
        )
        return GCSPublishResult(
            status="verified",
            dataset_id=str(manifest_payload["dataset_id"]),
            market_data_as_of=validation.last_date,
            bucket=self.bucket,
            prefix=self.prefix,
            parquet_uri=str(parquet_descriptor["gs_uri"]),
            metadata_uri=str(metadata_descriptor["gs_uri"]),
            manifest_uri=f"gs://{self.bucket}/{manifest_object}",
            parquet_sha256=str(parquet_descriptor["sha256"]),
            row_count=validation.row_count,
            symbols=list(symbols),
            first_date=validation.first_date,
            last_date=validation.last_date,
            verified=True,
            warnings=list(validation.warnings),
        )

    def _build_result(
        self,
        *,
        status: str,
        local: _LocalDatasetContext,
        parquet_uri: str,
        metadata_uri: str,
        manifest_uri: str,
        uploaded_objects: list[str] | None = None,
        reused_objects: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> GCSPublishResult:
        return GCSPublishResult(
            status=status,
            dataset_id=local.dataset_id,
            market_data_as_of=local.validation.last_date,
            bucket=self.bucket,
            prefix=self.prefix,
            parquet_uri=parquet_uri,
            metadata_uri=metadata_uri,
            manifest_uri=manifest_uri,
            parquet_sha256=local.parquet_sha256,
            row_count=local.validation.row_count,
            symbols=list(self.store.symbols),
            first_date=local.validation.first_date,
            last_date=local.validation.last_date,
            uploaded_objects=list(uploaded_objects or []),
            reused_objects=list(reused_objects or []),
            verified=True,
            warnings=list(warnings or []),
        )

    @staticmethod
    def _manifest_matches_local(
        *,
        manifest_payload: dict[str, Any],
        dataset_id: str,
        parquet_sha256: str,
        metadata_sha256: str,
    ) -> bool:
        return (
            str(manifest_payload.get("dataset_id", "")).strip() == dataset_id
            and str(dict(manifest_payload.get("parquet", {})).get("sha256", "")).strip() == parquet_sha256
            and str(dict(manifest_payload.get("metadata", {})).get("sha256", "")).strip() == metadata_sha256
        )

    def _load_local_dataset(self) -> _LocalDatasetContext:
        frame = self.store.load_offline()
        validation = self.store.validate_frame(frame)
        metadata_payload = self.store.load_metadata()
        _validate_local_metadata(store=self.store, validation=validation, metadata_payload=metadata_payload)
        parquet_bytes = Path(self.store.parquet_path).read_bytes()
        metadata_bytes = Path(self.store.metadata_path).read_bytes()
        parquet_sha256 = _parquet_file_sha256(self.store.parquet_path)
        metadata_sha256 = _sha256_bytes(metadata_bytes)
        dataset_id = _dataset_id(market_data_as_of=validation.last_date, parquet_sha256=parquet_sha256)
        return _LocalDatasetContext(
            store=self.store,
            validation=validation,
            metadata_payload=metadata_payload,
            parquet_bytes=parquet_bytes,
            metadata_bytes=metadata_bytes,
            parquet_sha256=parquet_sha256,
            metadata_sha256=metadata_sha256,
            dataset_id=dataset_id,
        )

    def _publish_immutable_object(
        self,
        *,
        object_name: str,
        payload: bytes,
        sha256: str,
        content_type: str,
    ) -> tuple[GCSObjectDescriptor, bool]:
        existing = self.storage_client.get_object_metadata(self.bucket, object_name)
        if existing is None:
            try:
                created = self.storage_client.upload_bytes(
                    bucket=self.bucket,
                    object_name=object_name,
                    data=payload,
                    if_generation_match=0,
                    content_type=content_type,
                )
                return self._build_descriptor(created, sha256), True
            except StorageManifestConflictError:
                existing = self.storage_client.get_object_metadata(self.bucket, object_name)
                if existing is None:
                    raise StorageObjectConflictError(f"Immutable object creation conflicted for `{object_name}`.")
            except StorageError:
                raise
        existing_bytes = self.storage_client.download_bytes(self.bucket, object_name)
        existing_sha256 = _sha256_bytes(existing_bytes)
        if existing_sha256 != sha256:
            raise StorageObjectConflictError(
                f"Immutable object `{object_name}` already exists with different content."
            )
        if existing is None:
            existing = self.storage_client.get_object_metadata(self.bucket, object_name)
            if existing is None:
                raise StorageNotFoundError(f"Immutable object `{object_name}` could not be reloaded after upload.")
        return self._build_descriptor(existing, sha256), False

    @staticmethod
    def _build_descriptor(metadata: StorageObjectMetadata, sha256: str) -> GCSObjectDescriptor:
        return GCSObjectDescriptor(
            bucket=metadata.bucket,
            object_name=metadata.object_name,
            sha256=sha256,
            size_bytes=metadata.size_bytes,
            generation=metadata.generation,
        )

    def _verify_remote_bytes(self, *, object_name: str, expected_sha256: str) -> None:
        downloaded = self.storage_client.download_bytes(self.bucket, object_name)
        actual_sha256 = _sha256_bytes(downloaded)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Uploaded GCS object `{object_name}` failed SHA-256 verification.")

    def _read_existing_manifest(self, manifest_object: str) -> dict[str, Any] | None:
        metadata = self.storage_client.get_object_metadata(self.bucket, manifest_object)
        if metadata is None:
            return None
        manifest_bytes = self.storage_client.download_bytes(self.bucket, manifest_object)
        try:
            payload = json.loads(manifest_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Existing GCS latest manifest is malformed JSON.") from exc
        _validate_manifest_payload(payload)
        return {"metadata": metadata, "payload": payload}


def publish_sector_store_to_gcs(
    *,
    bucket: str,
    prefix: str = DEFAULT_GCS_PREFIX,
    project: str | None = None,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    dry_run: bool = False,
    storage_client: StorageClientProtocol | None = None,
) -> GCSPublishResult:
    store = SectorPriceStore(
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        symbols=symbols,
    )
    publisher = SectorHistoryGCSPublisher(
        store=store,
        storage_client=storage_client or GoogleCloudStorageClient(project=project),
        bucket=bucket,
        prefix=prefix,
    )
    return publisher.publish(dry_run=dry_run)


def verify_sector_store_in_gcs(
    *,
    bucket: str,
    prefix: str = DEFAULT_GCS_PREFIX,
    project: str | None = None,
    storage_client: StorageClientProtocol | None = None,
) -> GCSPublishResult:
    publisher = SectorHistoryGCSPublisher(
        store=SectorPriceStore(),
        storage_client=storage_client or GoogleCloudStorageClient(project=project),
        bucket=bucket,
        prefix=prefix,
    )
    return publisher.verify()
