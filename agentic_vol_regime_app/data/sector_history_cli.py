from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_cli_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "agentic_vol_regime_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_cli_imports()

from agentic_vol_regime_app.data.sector_history_gcs import (  # noqa: E402
    DEFAULT_GCS_PREFIX,
    publish_sector_store_to_gcs,
    verify_sector_store_in_gcs,
)
from agentic_vol_regime_app.data.sector_history_store import (  # noqa: E402
    DEFAULT_SECTOR_PRICE_SYMBOLS,
    SectorHistorySyncResult,
    SectorPriceStore,
    sync_sector_history,
)
from agentic_vol_regime_app.data.sector_history_update_publish import (  # noqa: E402
    SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES,
    update_and_publish_sector_history,
)
from agentic_vol_regime_app.data.vol_regime_history_gcs import (  # noqa: E402
    DEFAULT_VOL_REGIME_GCS_BUCKET,
    DEFAULT_VOL_REGIME_GCS_PREFIX,
    sync_and_publish_vol_regime_history,
)


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _shared_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", default=None, help="Optional explicit parquet output path."
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional explicit metadata JSON output path.",
    )
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbol universe."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=73)
    parser.add_argument("--readonly", action="store_true", default=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--market-data-type", type=int, default=1)
    parser.add_argument("--preferred-what-to-show", default="ADJUSTED_LAST")
    parser.add_argument("--no-trades-fallback", action="store_true")


def _render_sync_result(result: SectorHistorySyncResult) -> str:
    lines = [
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        "",
        "Per-symbol summary:",
    ]
    for symbol, payload in result.symbols.items():
        lines.append(
            f"{symbol}: status={payload.status} prev={payload.previous_last_date} "
            f"request={payload.requested_start}->{payload.requested_end} "
            f"received={payload.received} inserted={payload.inserted} revised={payload.revised} "
            f"new_last={payload.new_last_date} whatToShow={payload.actual_what_to_show}"
        )
    return "\n".join(lines)


def _render_frame_summary(
    frame: pd.DataFrame, *, store: SectorPriceStore, include_rows: bool
) -> str:
    validation = store.validate_frame(frame)
    payload: dict[str, Any] = {
        "schema_version": "sector_prices.v1",
        "row_count": validation.row_count,
        "first_date": validation.first_date,
        "last_date": validation.last_date,
        "symbols": list(store.symbols),
        "per_symbol": validation.per_symbol,
    }
    if include_rows:
        rows = frame.copy()
        rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
        payload["rows"] = rows.tail(10).to_dict(orient="records")
    return json.dumps(payload, indent=2, sort_keys=True)


def _render_publish_result(result: Any) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)


def _exit_code_for_update_publish_status(status: str) -> int:
    return 0 if status in SUCCESSFUL_UPDATE_AND_PUBLISH_STATUSES else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the offline-first IBKR sector price history store."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Create or rebuild the authoritative sector history store."
    )
    _shared_parser(bootstrap)
    bootstrap.add_argument("--start-date", default=None)
    bootstrap.add_argument("--history-years", type=int, default=None)
    bootstrap.add_argument("--history-days", type=int, default=None)
    bootstrap.add_argument("--target-end-date", default=None)
    bootstrap.add_argument("--force", action="store_true")

    update = subparsers.add_parser(
        "update", help="Fetch only the missing delta plus overlap."
    )
    _shared_parser(update)
    update.add_argument("--target-end-date", default=None)
    update.add_argument("--overlap-trading-days", type=int, default=5)
    update.add_argument("--repair-start-date", default=None)
    update.add_argument("--bootstrap-start-date", default=None)

    offline = subparsers.add_parser(
        "offline", help="Load and validate the local store without IBKR."
    )
    offline.add_argument("--output", default=None)
    offline.add_argument("--metadata-output", default=None)
    offline.add_argument("--symbols", default=None)
    offline.add_argument("--summary", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Validate the local store and print a summary."
    )
    validate.add_argument("--output", default=None)
    validate.add_argument("--metadata-output", default=None)
    validate.add_argument("--symbols", default=None)

    publish_gcs = subparsers.add_parser(
        "publish-gcs",
        help="Validate the local store and publish immutable dataset objects plus the latest GCS manifest.",
    )
    publish_gcs.add_argument("--output", default=None)
    publish_gcs.add_argument("--metadata-output", default=None)
    publish_gcs.add_argument("--symbols", default=None)
    publish_gcs.add_argument(
        "--bucket", default=os.getenv("MARKET_MANIFOLD_GCS_BUCKET")
    )
    publish_gcs.add_argument(
        "--prefix", default=os.getenv("MARKET_MANIFOLD_GCS_PREFIX", DEFAULT_GCS_PREFIX)
    )
    publish_gcs.add_argument(
        "--project", default=os.getenv("MARKET_MANIFOLD_GCP_PROJECT")
    )
    publish_gcs.add_argument("--dry-run", action="store_true")

    verify_gcs = subparsers.add_parser(
        "verify-gcs",
        help="Read GCS latest.json, verify immutable objects, and validate the referenced parquet store.",
    )
    verify_gcs.add_argument("--bucket", default=os.getenv("MARKET_MANIFOLD_GCS_BUCKET"))
    verify_gcs.add_argument(
        "--prefix", default=os.getenv("MARKET_MANIFOLD_GCS_PREFIX", DEFAULT_GCS_PREFIX)
    )
    verify_gcs.add_argument(
        "--project", default=os.getenv("MARKET_MANIFOLD_GCP_PROJECT")
    )

    update_publish_gcs = subparsers.add_parser(
        "update-and-publish-gcs",
        help="Update the local sector store from IBKR, validate publication readiness, then publish and verify in GCS.",
    )
    _shared_parser(update_publish_gcs)
    update_publish_gcs.add_argument("--target-end-date", default=None)
    update_publish_gcs.add_argument("--overlap-trading-days", type=int, default=5)
    update_publish_gcs.add_argument("--repair-start-date", default=None)
    update_publish_gcs.add_argument("--bootstrap-start-date", default=None)
    update_publish_gcs.add_argument(
        "--bucket", default=os.getenv("MARKET_MANIFOLD_GCS_BUCKET")
    )
    update_publish_gcs.add_argument(
        "--prefix", default=os.getenv("MARKET_MANIFOLD_GCS_PREFIX", DEFAULT_GCS_PREFIX)
    )
    update_publish_gcs.add_argument(
        "--project", default=os.getenv("MARKET_MANIFOLD_GCP_PROJECT")
    )
    update_publish_gcs.add_argument("--require-adjusted-last", action="store_true")
    update_publish_gcs.add_argument(
        "--skip-publish-if-already-current", action="store_true"
    )
    update_publish_gcs.add_argument("--dry-run-publish", action="store_true")

    vol_regime_gcs = subparsers.add_parser(
        "sync-vol-regime-history-gcs",
        help="Sync IBKR SPY/VIX/VVIX history, then publish and verify it in GCS.",
    )
    vol_regime_gcs.add_argument("--output", default=None, help="Optional explicit parquet output path.")
    vol_regime_gcs.add_argument("--metadata-output", default=None, help="Optional explicit metadata JSON output path.")
    vol_regime_gcs.add_argument("--host", default="127.0.0.1")
    vol_regime_gcs.add_argument("--port", type=int, default=4001)
    vol_regime_gcs.add_argument("--client-id", type=int, default=73)
    vol_regime_gcs.add_argument("--readonly", action="store_true", default=True)
    vol_regime_gcs.add_argument("--timeout-seconds", type=float, default=10.0)
    vol_regime_gcs.add_argument("--market-data-type", type=int, default=1)
    vol_regime_gcs.add_argument("--target-end-date", default=None)
    vol_regime_gcs.add_argument("--bootstrap-start-date", default=None)
    vol_regime_gcs.add_argument("--history-years", type=int, default=None)
    vol_regime_gcs.add_argument("--history-days", type=int, default=None)
    vol_regime_gcs.add_argument("--overlap-trading-days", type=int, default=5)
    vol_regime_gcs.add_argument("--bucket", default=DEFAULT_VOL_REGIME_GCS_BUCKET)
    vol_regime_gcs.add_argument("--prefix", default=os.getenv("MARKET_MANIFOLD_VOL_REGIME_GCS_PREFIX", DEFAULT_VOL_REGIME_GCS_PREFIX))
    vol_regime_gcs.add_argument("--project", default=os.getenv("MARKET_MANIFOLD_GCP_PROJECT", "marketphysics"))
    vol_regime_gcs.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if (
        args.command in {"publish-gcs", "verify-gcs", "update-and-publish-gcs", "sync-vol-regime-history-gcs"}
        and not args.bucket
    ):
        parser.error("--bucket is required or set MARKET_MANIFOLD_GCS_BUCKET.")

    store = SectorPriceStore(
        parquet_path=getattr(args, "output", None),
        metadata_path=getattr(args, "metadata_output", None),
        symbols=_parse_symbols(getattr(args, "symbols", None))
        or list(DEFAULT_SECTOR_PRICE_SYMBOLS),
    )

    if args.command == "sync-vol-regime-history-gcs":
        result = sync_and_publish_vol_regime_history(
            bucket=args.bucket,
            prefix=args.prefix,
            project=args.project,
            parquet_path=args.output,
            metadata_path=args.metadata_output,
            target_end_date=args.target_end_date,
            bootstrap_start_date=args.bootstrap_start_date,
            history_years=args.history_years,
            history_days=args.history_days,
            overlap_trading_days=args.overlap_trading_days,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            readonly=args.readonly,
            timeout_seconds=args.timeout_seconds,
            market_data_type=args.market_data_type,
            dry_run=bool(args.dry_run),
        )
        print(_render_publish_result(result))
        return

    if args.command == "bootstrap":
        result = sync_sector_history(
            mode="bootstrap",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            bootstrap_start_date=args.start_date,
            history_years=args.history_years,
            history_days=args.history_days,
            target_end_date=args.target_end_date,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            readonly=args.readonly,
            timeout_seconds=args.timeout_seconds,
            market_data_type=args.market_data_type,
            preferred_what_to_show=args.preferred_what_to_show,
            allow_trades_fallback=not args.no_trades_fallback,
            force=args.force,
        )
        print(_render_sync_result(result))
        return

    if args.command == "update":
        result = sync_sector_history(
            mode="update",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            target_end_date=args.target_end_date,
            bootstrap_start_date=args.bootstrap_start_date,
            overlap_trading_days=args.overlap_trading_days,
            repair_start_date=args.repair_start_date,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            readonly=args.readonly,
            timeout_seconds=args.timeout_seconds,
            market_data_type=args.market_data_type,
            preferred_what_to_show=args.preferred_what_to_show,
            allow_trades_fallback=not args.no_trades_fallback,
        )
        print(_render_sync_result(result))
        return

    if args.command == "offline":
        frame = sync_sector_history(
            mode="offline",
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
        )
        print(
            _render_frame_summary(frame, store=store, include_rows=bool(args.summary))
        )
        return

    if args.command == "publish-gcs":
        result = publish_sector_store_to_gcs(
            bucket=args.bucket,
            prefix=args.prefix,
            project=args.project,
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            dry_run=bool(args.dry_run),
        )
        print(_render_publish_result(result))
        return

    if args.command == "verify-gcs":
        result = verify_sector_store_in_gcs(
            bucket=args.bucket,
            prefix=args.prefix,
            project=args.project,
        )
        print(_render_publish_result(result))
        return

    if args.command == "update-and-publish-gcs":
        result = update_and_publish_sector_history(
            bucket=args.bucket,
            prefix=args.prefix,
            project=args.project,
            parquet_path=store.parquet_path,
            metadata_path=store.metadata_path,
            symbols=list(store.symbols),
            target_end_date=args.target_end_date,
            overlap_trading_days=args.overlap_trading_days,
            repair_start_date=args.repair_start_date,
            bootstrap_start_date=args.bootstrap_start_date,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            readonly=args.readonly,
            timeout_seconds=args.timeout_seconds,
            market_data_type=args.market_data_type,
            preferred_what_to_show=args.preferred_what_to_show,
            allow_trades_fallback=not args.no_trades_fallback,
            publish_if_already_current=not args.skip_publish_if_already_current,
            require_adjusted_last=bool(args.require_adjusted_last),
            dry_run_publish=bool(args.dry_run_publish),
        )
        print(_render_publish_result(result))
        raise SystemExit(_exit_code_for_update_publish_status(result.status))

    validation = sync_sector_history(
        mode="validate",
        parquet_path=store.parquet_path,
        metadata_path=store.metadata_path,
        symbols=list(store.symbols),
    )
    print(
        json.dumps(
            {
                "schema_version": "sector_prices.v1",
                "row_count": validation.row_count,
                "first_date": validation.first_date,
                "last_date": validation.last_date,
                "per_symbol": validation.per_symbol,
                "warnings": validation.warnings,
                "parquet_path": str(store.parquet_path),
                "metadata_path": str(store.metadata_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
