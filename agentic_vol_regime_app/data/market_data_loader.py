"""Market snapshot loading for the volatility regime app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_vol_regime_app.config import load_json
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKROptionChainRequest,
)


def load_market_snapshot(
    input_payload: dict[str, Any],
    *,
    app_root: Path,
    data_pipe: IBKRDataPipe | None = None,
) -> ObservationRecord:
    """Load a normalized observation record from inline input or JSON."""
    snapshot = input_payload.get("market_snapshot")
    if snapshot is None:
        provider = str(input_payload.get("data_provider", "")).strip().lower()
        if provider == "ibkr":
            ibkr_payload = dict(input_payload.get("ibkr", {}))
            request = IBKROptionChainRequest.from_payload(
                {
                    "symbol": input_payload.get("symbol", ibkr_payload.get("symbol", "SPY")),
                    **ibkr_payload,
                }
            )
            pipe = data_pipe or IBKRDataPipe(
                connection=IBKRConnectionConfig(
                    host=str(ibkr_payload.get("host", "127.0.0.1")),
                    port=int(ibkr_payload.get("port", 4001)),
                    client_id=int(ibkr_payload.get("client_id", 73)),
                    readonly=bool(ibkr_payload.get("readonly", True)),
                    timeout_seconds=float(ibkr_payload.get("timeout_seconds", 10.0)),
                    market_data_type=int(ibkr_payload.get("market_data_type", 1)),
                )
            )
            return pipe.fetch_market_snapshot(request)
        snapshot_path = input_payload.get("snapshot_path")
        if not snapshot_path:
            raise ValueError(
                "Expected 'market_snapshot', 'snapshot_path', or 'data_provider=ibkr' in input payload."
            )
        resolved_path = Path(str(snapshot_path))
        if not resolved_path.is_absolute():
            resolved_path = (Path.cwd() / resolved_path).resolve()
            if not resolved_path.exists():
                resolved_path = (app_root / str(snapshot_path)).resolve()
        snapshot = load_json(resolved_path)

    symbols = dict(snapshot.get("symbols", {}))
    if not symbols:
        raise ValueError("Market snapshot must include a non-empty 'symbols' mapping.")

    return ObservationRecord(
        schema_version=str(snapshot.get("schema_version", "observation.v1")),
        as_of=str(snapshot["as_of"]),
        source=str(snapshot.get("source", "input")),
        symbols=symbols,
        history={key: list(value) for key, value in dict(snapshot.get("history", {})).items()},
        quality=dict(snapshot.get("quality", {})),
        option_chain=dict(snapshot.get("option_chain", {})),
        provider_metadata=dict(snapshot.get("provider_metadata", {})),
    )
