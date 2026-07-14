from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from agentic_vol_regime_app.data.ibkr_client import IBKRDataPipe
from agentic_vol_regime_app.data.sector_history_gcs import (
    DEFAULT_GCS_PREFIX,
    GCSPublishResult,
    StorageClientProtocol,
    _validate_local_metadata,
    publish_sector_store_to_gcs,
    verify_sector_store_in_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import (
    DEFAULT_SECTOR_PRICE_SYMBOLS,
    SectorHistorySyncResult,
    SectorPriceStore,
    resolve_target_completed_session,
    sync_sector_history,
)


UPDATE_AND_PUBLISH_SCHEMA_VERSION = "sector_history_update_publish.v1"
SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES = {
    "published",
    "already_published",
    "already_current_publish_skipped",
    "dry_run_completed",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, allow_nan=False))


def _optional_symbols_from_metadata(metadata_payload: dict[str, Any]) -> set[str]:
    optional = {
        str(item).strip().upper()
        for item in metadata_payload.get("optional_symbols", [])
        if str(item).strip()
    }
    for symbol, entry in dict(metadata_payload.get("per_symbol", {})).items():
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue
        values = dict(entry)
        if values.get("optional") is True or values.get("required") is False:
            optional.add(normalized)
    return optional


def _fallback_warnings_from_metadata(
    *, store: SectorPriceStore, metadata_payload: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    per_symbol_metadata = {
        str(symbol).strip().upper(): dict(values)
        for symbol, values in dict(metadata_payload.get("per_symbol", {})).items()
    }
    for symbol in store.symbols:
        actual_what_to_show = (
            str(per_symbol_metadata.get(symbol, {}).get("actual_what_to_show") or "")
            .strip()
            .upper()
        )
        if actual_what_to_show in {"TRADES", "MIXED"}:
            warnings.append(
                f"{symbol} publication is using {actual_what_to_show} data; prices are not labeled adjusted."
            )
    return warnings


def _local_update_status(result: SectorHistorySyncResult) -> str:
    if any(
        symbol_result.status == "updated" for symbol_result in result.symbols.values()
    ):
        return "updated"
    return "already_current"


@dataclass(slots=True)
class PublicationReadinessResult:
    ready: bool
    required_symbols: list[str]
    current_symbols: list[str]
    lagging_symbols: dict[str, dict[str, Any]]
    blocked_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": bool(self.ready),
            "required_symbols": list(self.required_symbols),
            "current_symbols": list(self.current_symbols),
            "lagging_symbols": dict(self.lagging_symbols),
            "blocked_reasons": list(self.blocked_reasons),
        }


@dataclass(slots=True)
class LocalUpdateSummary:
    status: str
    ibkr_request_count: int
    store_before: dict[str, Any]
    store_after: dict[str, Any]
    symbols: dict[str, dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ibkr_request_count": int(self.ibkr_request_count),
            "store_before": dict(self.store_before),
            "store_after": dict(self.store_after),
            "symbols": dict(self.symbols),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class GCSPublishSummary:
    attempted: bool
    status: str
    dataset_id: str | None = None
    manifest_uri: str | None = None
    parquet_uri: str | None = None
    parquet_sha256: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": bool(self.attempted),
            "status": self.status,
            "dataset_id": self.dataset_id,
            "manifest_uri": self.manifest_uri,
            "parquet_uri": self.parquet_uri,
            "parquet_sha256": self.parquet_sha256,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class GCSVerifySummary:
    attempted: bool
    verified: bool
    dataset_id: str | None = None
    parquet_sha256: str | None = None
    status: str = "skipped"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": bool(self.attempted),
            "verified": bool(self.verified),
            "dataset_id": self.dataset_id,
            "parquet_sha256": self.parquet_sha256,
            "status": self.status,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class UpdateAndPublishResult:
    schema_version: str
    status: str
    target_completed_session: str | None
    started_at: str
    completed_at: str
    local_update: LocalUpdateSummary
    publication_readiness: PublicationReadinessResult
    gcs_publish: GCSPublishSummary
    gcs_verify: GCSVerifySummary
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": self.schema_version,
                "status": self.status,
                "target_completed_session": self.target_completed_session,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "local_update": self.local_update.to_dict(),
                "publication_readiness": self.publication_readiness.to_dict(),
                "gcs_publish": self.gcs_publish.to_dict(),
                "gcs_verify": self.gcs_verify.to_dict(),
                "warnings": list(self.warnings),
                "errors": list(self.errors),
            }
        )


def _empty_result(
    *,
    status: str,
    target_completed_session: str | None,
    started_at: str,
    errors: list[str],
) -> UpdateAndPublishResult:
    completed_at = _utc_now_iso()
    return UpdateAndPublishResult(
        schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
        status=status,
        target_completed_session=target_completed_session,
        started_at=started_at,
        completed_at=completed_at,
        local_update=LocalUpdateSummary(
            status="not_started",
            ibkr_request_count=0,
            store_before={},
            store_after={},
            symbols={},
            warnings=[],
        ),
        publication_readiness=PublicationReadinessResult(
            ready=False,
            required_symbols=[],
            current_symbols=[],
            lagging_symbols={},
            blocked_reasons=[],
        ),
        gcs_publish=GCSPublishSummary(
            attempted=False,
            status="not_attempted",
        ),
        gcs_verify=GCSVerifySummary(
            attempted=False,
            verified=False,
            status="not_attempted",
        ),
        errors=list(errors),
    )


def _evaluate_publication_readiness(
    *,
    store: SectorPriceStore,
    metadata_payload: dict[str, Any],
    target_completed_session: date,
    require_adjusted_last: bool,
) -> PublicationReadinessResult:
    per_symbol_metadata = {
        str(symbol).strip().upper(): dict(values)
        for symbol, values in dict(metadata_payload.get("per_symbol", {})).items()
    }
    optional_symbols = _optional_symbols_from_metadata(metadata_payload)
    required_symbols = [
        symbol for symbol in store.symbols if symbol not in optional_symbols
    ]
    lagging_symbols: dict[str, dict[str, Any]] = {}
    blocked_reasons: list[str] = []
    current_symbols: list[str] = []
    target_iso = target_completed_session.isoformat()

    for symbol in required_symbols:
        symbol_metadata = per_symbol_metadata.get(symbol, {})
        last_valid_date = (
            str(symbol_metadata.get("last_valid_date") or "").strip() or None
        )
        fetch_status = str(symbol_metadata.get("fetch_status") or "").strip() or None
        actual_what_to_show = (
            str(symbol_metadata.get("actual_what_to_show") or "").strip() or None
        )
        symbol_warnings = [
            str(item)
            for item in metadata_payload.get("warnings", [])
            if str(item).startswith(f"{symbol} ")
        ]
        internal_gap_count = int(symbol_metadata.get("internal_gap_count", 0) or 0)

        if last_valid_date == target_iso:
            current_symbols.append(symbol)
        else:
            lagging_symbols[symbol] = {
                "expected_last_date": target_iso,
                "actual_last_date": last_valid_date,
                "fetch_status": fetch_status,
                "actual_what_to_show": actual_what_to_show,
                "internal_gap_count": internal_gap_count,
                "warnings": symbol_warnings,
            }
            blocked_reasons.append(
                f"{symbol} is behind target {target_iso}; last valid date is {last_valid_date or 'missing'}."
            )

        if require_adjusted_last and actual_what_to_show in {"TRADES", "MIXED"}:
            blocked_reasons.append(
                f"{symbol} used {actual_what_to_show} instead of ADJUSTED_LAST and strict adjusted-only mode is enabled."
            )

    return PublicationReadinessResult(
        ready=not blocked_reasons,
        required_symbols=required_symbols,
        current_symbols=current_symbols,
        lagging_symbols=lagging_symbols,
        blocked_reasons=blocked_reasons,
    )


def _publish_summary_from_result(result: GCSPublishResult) -> GCSPublishSummary:
    return GCSPublishSummary(
        attempted=True,
        status=result.status,
        dataset_id=result.dataset_id,
        manifest_uri=result.manifest_uri,
        parquet_uri=result.parquet_uri,
        parquet_sha256=result.parquet_sha256,
        warnings=list(result.warnings),
    )


def _verify_summary_from_result(result: GCSPublishResult) -> GCSVerifySummary:
    return GCSVerifySummary(
        attempted=True,
        verified=bool(result.verified),
        dataset_id=result.dataset_id,
        parquet_sha256=result.parquet_sha256,
        status=result.status,
        warnings=list(result.warnings),
    )


def update_and_publish_sector_history(
    *,
    bucket: str,
    prefix: str = DEFAULT_GCS_PREFIX,
    project: str | None = None,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
    target_end_date: date | str | None = None,
    overlap_trading_days: int = 5,
    repair_start_date: date | str | None = None,
    bootstrap_start_date: date | str | None = None,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    readonly: bool = True,
    timeout_seconds: float = 10.0,
    market_data_type: int = 1,
    preferred_what_to_show: str = "ADJUSTED_LAST",
    allow_trades_fallback: bool = True,
    require_all_symbols_current: bool = True,
    publish_if_already_current: bool = True,
    require_adjusted_last: bool = False,
    dry_run_publish: bool = False,
    data_pipe: IBKRDataPipe | None = None,
    storage_client: StorageClientProtocol | None = None,
) -> UpdateAndPublishResult:
    started_at = _utc_now_iso()
    normalized_bucket = str(bucket or "").strip()
    normalized_prefix = str(prefix or "").strip().strip("/")
    store = SectorPriceStore(
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        symbols=symbols or list(DEFAULT_SECTOR_PRICE_SYMBOLS),
    )

    if not normalized_bucket:
        return _empty_result(
            status="invalid_configuration",
            target_completed_session=None,
            started_at=started_at,
            errors=[
                "GCS bucket is required. Supply --bucket or MARKET_MANIFOLD_GCS_BUCKET before running the update."
            ],
        )
    if not normalized_prefix:
        return _empty_result(
            status="invalid_configuration",
            target_completed_session=None,
            started_at=started_at,
            errors=["GCS prefix must be non-empty."],
        )
    if not store.symbols:
        return _empty_result(
            status="invalid_configuration",
            target_completed_session=None,
            started_at=started_at,
            errors=["Sector history symbol universe is empty."],
        )
    if not store.exists():
        return _empty_result(
            status="invalid_configuration",
            target_completed_session=None,
            started_at=started_at,
            errors=[
                "Local sector history store does not exist. Bootstrap the authoritative parquet store first with the existing bootstrap command."
            ],
        )
    existing_summary = store.summarize_existing_store()
    if int(existing_summary.get("row_count", 0) or 0) <= 0:
        return _empty_result(
            status="invalid_configuration",
            target_completed_session=None,
            started_at=started_at,
            errors=[
                "Local sector history store is empty. Bootstrap the authoritative parquet store first."
            ],
        )

    target_completed_session = resolve_target_completed_session(
        explicit_target_end_date=target_end_date
    )
    target_iso = target_completed_session.isoformat()

    try:
        sync_result = sync_sector_history(
            mode="update",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            target_end_date=target_end_date,
            bootstrap_start_date=bootstrap_start_date,
            overlap_trading_days=overlap_trading_days,
            repair_start_date=repair_start_date,
            host=host,
            port=port,
            client_id=client_id,
            readonly=readonly,
            timeout_seconds=timeout_seconds,
            market_data_type=market_data_type,
            preferred_what_to_show=preferred_what_to_show,
            allow_trades_fallback=allow_trades_fallback,
            data_pipe=data_pipe,
        )
        if not isinstance(sync_result, SectorHistorySyncResult):
            raise TypeError("Sector history update returned an unexpected result type.")
        if sync_result.mode != "update":
            raise ValueError(
                f"Unexpected sector history mode returned from updater: {sync_result.mode}"
            )
        if any(
            symbol_result.status == "bootstrapped"
            for symbol_result in sync_result.symbols.values()
        ):
            raise ValueError(
                "Update orchestration refused to continue because the updater reported a bootstrap result."
            )
    except Exception as exc:
        result = _empty_result(
            status="update_failed",
            target_completed_session=target_iso,
            started_at=started_at,
            errors=[str(exc)],
        )
        result.local_update = LocalUpdateSummary(
            status="failed",
            ibkr_request_count=0,
            store_before=existing_summary,
            store_after=store.summarize_existing_store(),
            symbols={},
            warnings=[],
        )
        return result

    local_update = LocalUpdateSummary(
        status=_local_update_status(sync_result),
        ibkr_request_count=sync_result.ibkr_request_count,
        store_before=dict(sync_result.store_before),
        store_after=dict(sync_result.store_after),
        symbols={
            symbol: payload.to_dict() for symbol, payload in sync_result.symbols.items()
        },
        warnings=list(sync_result.warnings),
    )

    try:
        frame = store.load_offline()
        validation = store.validate_frame(frame)
        metadata_payload = store.load_metadata()
        _validate_local_metadata(
            store=store, validation=validation, metadata_payload=metadata_payload
        )
    except Exception as exc:
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="local_validation_failed",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=PublicationReadinessResult(
                ready=False,
                required_symbols=list(store.symbols),
                current_symbols=[],
                lagging_symbols={},
                blocked_reasons=["Local validation failed before GCS publication."],
            ),
            gcs_publish=GCSPublishSummary(attempted=False, status="skipped"),
            gcs_verify=GCSVerifySummary(
                attempted=False, verified=False, status="skipped"
            ),
            warnings=list(sync_result.warnings),
            errors=[str(exc)],
        )

    readiness = _evaluate_publication_readiness(
        store=store,
        metadata_payload=metadata_payload,
        target_completed_session=target_completed_session,
        require_adjusted_last=require_adjusted_last,
    )
    metadata_fallback_warnings = _fallback_warnings_from_metadata(
        store=store, metadata_payload=metadata_payload
    )
    if not require_all_symbols_current and not readiness.ready:
        readiness = PublicationReadinessResult(
            ready=True,
            required_symbols=readiness.required_symbols,
            current_symbols=readiness.current_symbols,
            lagging_symbols=readiness.lagging_symbols,
            blocked_reasons=[],
        )

    if not readiness.ready:
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="update_incomplete_not_published",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=readiness,
            gcs_publish=GCSPublishSummary(attempted=False, status="skipped"),
            gcs_verify=GCSVerifySummary(
                attempted=False, verified=False, status="skipped"
            ),
            warnings=list(sync_result.warnings)
            + list(validation.warnings)
            + metadata_fallback_warnings,
            errors=[],
        )

    if local_update.status == "already_current" and not publish_if_already_current:
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="already_current_publish_skipped",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=readiness,
            gcs_publish=GCSPublishSummary(attempted=False, status="skipped"),
            gcs_verify=GCSVerifySummary(
                attempted=False, verified=False, status="skipped"
            ),
            warnings=list(sync_result.warnings)
            + list(validation.warnings)
            + metadata_fallback_warnings,
            errors=[],
        )

    try:
        publish_result = publish_sector_store_to_gcs(
            bucket=normalized_bucket,
            prefix=normalized_prefix,
            project=project,
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            dry_run=bool(dry_run_publish),
            storage_client=storage_client,
        )
    except Exception as exc:
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="local_updated_cloud_publish_failed",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=readiness,
            gcs_publish=GCSPublishSummary(attempted=True, status="failed"),
            gcs_verify=GCSVerifySummary(
                attempted=False, verified=False, status="skipped"
            ),
            warnings=list(sync_result.warnings)
            + list(validation.warnings)
            + metadata_fallback_warnings,
            errors=[
                str(exc),
                "The local parquet update was retained. Rerunning will reuse the current local store and should not need a full IBKR redownload.",
            ],
        )

    publish_summary = _publish_summary_from_result(publish_result)
    combined_warnings = (
        list(sync_result.warnings)
        + list(validation.warnings)
        + metadata_fallback_warnings
        + list(publish_result.warnings)
    )

    if dry_run_publish:
        combined_warnings.append(
            "Dry-run publication simulated the GCS publish step. Local data may have changed; no remote verification was performed."
        )
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="dry_run_completed",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=readiness,
            gcs_publish=publish_summary,
            gcs_verify=GCSVerifySummary(
                attempted=False, verified=False, status="skipped"
            ),
            warnings=combined_warnings,
            errors=[],
        )

    try:
        verify_result = verify_sector_store_in_gcs(
            bucket=normalized_bucket,
            prefix=normalized_prefix,
            project=project,
            storage_client=storage_client,
        )
    except Exception as exc:
        return UpdateAndPublishResult(
            schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
            status="verification_failed",
            target_completed_session=target_iso,
            started_at=started_at,
            completed_at=_utc_now_iso(),
            local_update=local_update,
            publication_readiness=readiness,
            gcs_publish=publish_summary,
            gcs_verify=GCSVerifySummary(
                attempted=True, verified=False, status="failed"
            ),
            warnings=combined_warnings,
            errors=[str(exc)],
        )

    verify_summary = _verify_summary_from_result(verify_result)
    errors: list[str] = []
    if publish_result.dataset_id != verify_result.dataset_id:
        errors.append(
            f"GCS verification dataset_id mismatch: published={publish_result.dataset_id} verified={verify_result.dataset_id}."
        )
    if publish_result.parquet_sha256 != verify_result.parquet_sha256:
        errors.append(
            "GCS verification parquet checksum mismatch between publish result and verify result."
        )
    final_status = publish_result.status if not errors else "verification_failed"

    return UpdateAndPublishResult(
        schema_version=UPDATE_AND_PUBLISH_SCHEMA_VERSION,
        status=final_status,
        target_completed_session=target_iso,
        started_at=started_at,
        completed_at=_utc_now_iso(),
        local_update=local_update,
        publication_readiness=readiness,
        gcs_publish=publish_summary,
        gcs_verify=verify_summary,
        warnings=combined_warnings + list(verify_result.warnings),
        errors=errors,
    )
