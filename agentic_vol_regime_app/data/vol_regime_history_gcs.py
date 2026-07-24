"""Publish a small, validated IBKR SPY/VIX/VVIX history dataset to GCS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.data.ibkr_client import IBKRDataPipe
from agentic_vol_regime_app.data.sector_history_gcs import (
    GCSPublishResult,
    StorageClientProtocol,
    publish_sector_store_to_gcs,
    verify_sector_store_in_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import SectorHistorySyncResult, sync_sector_history


VOL_REGIME_HISTORY_SYMBOLS = ("SPY", "VIX", "VVIX")
DEFAULT_VOL_REGIME_GCS_BUCKET = "marketphysics-market-manifold-data"
DEFAULT_VOL_REGIME_GCS_PREFIX = "market-manifold/vol-regime-history"


def default_vol_regime_history_paths(app_paths: AppPaths | None = None) -> tuple[Path, Path]:
    root = (app_paths or AppPaths.default()).root / "data" / "market_history"
    return root / "vol_regime_prices_daily.parquet", root / "vol_regime_prices_daily.metadata.json"


@dataclass(frozen=True, slots=True)
class VolRegimeHistoryPublishResult:
    local_sync: SectorHistorySyncResult
    gcs_publish: GCSPublishResult
    gcs_verify: GCSPublishResult | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": "vol_regime_history.v1",
            "symbols": list(VOL_REGIME_HISTORY_SYMBOLS),
            "local_sync": self.local_sync.to_dict(),
            "gcs_publish": self.gcs_publish.to_dict(),
            "gcs_verify": self.gcs_verify.to_dict() if self.gcs_verify else None,
        }


def sync_and_publish_vol_regime_history(
    *,
    bucket: str = DEFAULT_VOL_REGIME_GCS_BUCKET,
    prefix: str = DEFAULT_VOL_REGIME_GCS_PREFIX,
    project: str | None = None,
    parquet_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    target_end_date: str | None = None,
    bootstrap_start_date: str | None = None,
    history_years: int | None = None,
    history_days: int | None = None,
    overlap_trading_days: int = 5,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    readonly: bool = True,
    timeout_seconds: float = 10.0,
    market_data_type: int = 1,
    dry_run: bool = False,
    data_pipe: IBKRDataPipe | None = None,
    storage_client: StorageClientProtocol | None = None,
) -> VolRegimeHistoryPublishResult:
    """Incrementally sync SPY/VIX/VVIX, then publish and verify the immutable dataset."""
    default_parquet, default_metadata = default_vol_regime_history_paths()
    resolved_parquet = Path(parquet_path) if parquet_path else default_parquet
    resolved_metadata = Path(metadata_path) if metadata_path else default_metadata
    mode = "update" if resolved_parquet.exists() else "bootstrap"
    sync_result = sync_sector_history(
        mode=mode,
        parquet_path=resolved_parquet,
        metadata_path=resolved_metadata,
        symbols=VOL_REGIME_HISTORY_SYMBOLS,
        target_end_date=target_end_date,
        bootstrap_start_date=bootstrap_start_date,
        history_years=history_years,
        history_days=history_days,
        overlap_trading_days=overlap_trading_days,
        host=host,
        port=port,
        client_id=client_id,
        readonly=readonly,
        timeout_seconds=timeout_seconds,
        market_data_type=market_data_type,
        data_pipe=data_pipe,
    )
    if not isinstance(sync_result, SectorHistorySyncResult):  # pragma: no cover - defensive contract guard
        raise TypeError("Volatility-regime history sync returned an unexpected result.")
    publish_result = publish_sector_store_to_gcs(
        bucket=bucket,
        prefix=prefix,
        project=project,
        parquet_path=resolved_parquet,
        metadata_path=resolved_metadata,
        symbols=VOL_REGIME_HISTORY_SYMBOLS,
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
    return VolRegimeHistoryPublishResult(
        local_sync=sync_result,
        gcs_publish=publish_result,
        gcs_verify=verify_result,
    )
