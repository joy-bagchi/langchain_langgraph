"""Market snapshot loading for the volatility regime app."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentic_vol_regime_app.config import load_json
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import (
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKROptionChainRequest,
)


def _parse_timestamp(value: Any) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_as_of_date(value: Any) -> date:
    text = str(value).strip()
    if not text:
        raise ValueError("Expected a non-empty as_of_date.")
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError("Expected as_of_date in YYYY-MM-DD format.") from exc


def _normalize_to_business_day(day: date) -> date:
    normalized = day
    while normalized.weekday() >= 5:
        normalized -= timedelta(days=1)
    return normalized


def _count_business_day_rewind(*, snapshot_day: date, requested_day: date) -> int:
    if requested_day > snapshot_day:
        raise ValueError(
            f"as_of_date {requested_day.isoformat()} is after snapshot as_of {snapshot_day.isoformat()}."
        )
    cursor = requested_day
    rewind_bars = 0
    while cursor < snapshot_day:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            rewind_bars += 1
    return rewind_bars


def _rewind_snapshot(snapshot: dict[str, Any], *, as_of_date: Any) -> dict[str, Any]:
    snapshot_ts = _parse_timestamp(snapshot["as_of"])
    requested_day = _normalize_to_business_day(_parse_as_of_date(as_of_date))
    snapshot_day = snapshot_ts.date()
    rewind_bars = _count_business_day_rewind(snapshot_day=snapshot_day, requested_day=requested_day)
    if rewind_bars == 0:
        return snapshot

    history = {
        str(key): [float(item) for item in list(values)]
        for key, values in dict(snapshot.get("history", {})).items()
        if isinstance(values, list)
    }
    if not history:
        raise ValueError("Cannot apply as_of_date without market history in the snapshot.")

    history_lengths = [len(values) for values in history.values() if values]
    if not history_lengths:
        raise ValueError("Cannot apply as_of_date because the snapshot history is empty.")
    max_rewind = min(history_lengths) - 1
    if rewind_bars > max_rewind:
        raise ValueError(
            f"as_of_date {requested_day.isoformat()} is outside the available history window for this snapshot."
        )

    rewound_history = {
        key: values[: len(values) - rewind_bars]
        for key, values in history.items()
    }

    symbols = {
        str(symbol): dict(payload)
        for symbol, payload in dict(snapshot.get("symbols", {})).items()
    }
    history_symbol_map = {
        "SPY_close": "SPY",
        "VIX": "VIX",
        "VVIX": "VVIX",
        "VIX9D": "VIX9D",
        "VIX3M": "VIX3M",
        "VIX6M": "VIX6M",
        "VIX9M": "VIX9M",
    }
    for history_key, symbol in history_symbol_map.items():
        values = rewound_history.get(history_key)
        if values and symbol in symbols:
            symbols[symbol]["last"] = float(values[-1])

    if "SPY_close" in rewound_history and rewound_history["SPY_close"]:
        spy_last = float(rewound_history["SPY_close"][-1])
        symbols.setdefault("SPY", {})["last"] = spy_last
        option_chain = dict(snapshot.get("option_chain", {}))
        if option_chain:
            option_chain["underlying_price"] = spy_last
            option_chain["fetched_at"] = snapshot_ts.replace(
                year=requested_day.year,
                month=requested_day.month,
                day=requested_day.day,
            ).isoformat().replace("+00:00", "Z")
            snapshot["option_chain"] = option_chain

    simulated_ts = snapshot_ts.replace(
        year=requested_day.year,
        month=requested_day.month,
        day=requested_day.day,
    )
    quality = dict(snapshot.get("quality", {}))
    warnings = [str(item) for item in list(quality.get("warnings", []))]
    warnings.append(
        f"Simulated as_of_date override applied: {requested_day.isoformat()} ({rewind_bars} business-day bars rewound)."
    )
    quality["warnings"] = warnings

    provider_metadata = dict(snapshot.get("provider_metadata", {}))
    provider_metadata["original_as_of"] = str(snapshot["as_of"])
    provider_metadata["simulated_as_of_date"] = requested_day.isoformat()
    provider_metadata["as_of_rewind_bars"] = rewind_bars

    snapshot["as_of"] = simulated_ts.isoformat().replace("+00:00", "Z")
    snapshot["symbols"] = symbols
    snapshot["history"] = rewound_history
    snapshot["quality"] = quality
    snapshot["provider_metadata"] = provider_metadata
    return snapshot


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
    snapshot = dict(snapshot)
    if input_payload.get("as_of_date"):
        snapshot = _rewind_snapshot(snapshot, as_of_date=input_payload["as_of_date"])

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
