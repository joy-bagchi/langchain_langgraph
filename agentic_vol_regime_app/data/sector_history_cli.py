from __future__ import annotations

import argparse
import json
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

from agentic_vol_regime_app.data.sector_history_store import (
    DEFAULT_SECTOR_PRICE_SYMBOLS,
    SectorHistorySyncResult,
    SectorPriceStore,
    sync_sector_history,
)


def _parse_symbols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _shared_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default=None, help="Optional explicit parquet output path.")
    parser.add_argument("--metadata-output", default=None, help="Optional explicit metadata JSON output path.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol universe.")
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


def _render_frame_summary(frame: pd.DataFrame, *, store: SectorPriceStore, include_rows: bool) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the offline-first IBKR sector price history store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create or rebuild the authoritative sector history store.")
    _shared_parser(bootstrap)
    bootstrap.add_argument("--start-date", default=None)
    bootstrap.add_argument("--history-years", type=int, default=None)
    bootstrap.add_argument("--history-days", type=int, default=None)
    bootstrap.add_argument("--target-end-date", default=None)
    bootstrap.add_argument("--force", action="store_true")

    update = subparsers.add_parser("update", help="Fetch only the missing delta plus overlap.")
    _shared_parser(update)
    update.add_argument("--target-end-date", default=None)
    update.add_argument("--overlap-trading-days", type=int, default=5)
    update.add_argument("--repair-start-date", default=None)
    update.add_argument("--bootstrap-start-date", default=None)

    offline = subparsers.add_parser("offline", help="Load and validate the local store without IBKR.")
    offline.add_argument("--output", default=None)
    offline.add_argument("--metadata-output", default=None)
    offline.add_argument("--symbols", default=None)
    offline.add_argument("--summary", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate the local store and print a summary.")
    validate.add_argument("--output", default=None)
    validate.add_argument("--metadata-output", default=None)
    validate.add_argument("--symbols", default=None)

    args = parser.parse_args()
    store = SectorPriceStore(
        parquet_path=args.output,
        metadata_path=args.metadata_output,
        symbols=_parse_symbols(args.symbols) or list(DEFAULT_SECTOR_PRICE_SYMBOLS),
    )

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
        print(_render_frame_summary(frame, store=store, include_rows=bool(args.summary)))
        return

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
