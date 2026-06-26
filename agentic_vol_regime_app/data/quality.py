"""Deterministic data quality checks."""

from __future__ import annotations

from typing import Any

from agentic_vol_regime_app.contracts import ObservationRecord


REQUIRED_SYMBOLS = ("SPY", "VIX", "VVIX", "VIX9D", "VIX3M")
REQUIRED_HISTORY = ("SPY_close", "VIX", "VVIX", "VIX9D", "VIX3M")


def validate_observation(
    observation: ObservationRecord,
    *,
    min_history_points: int = 22,
) -> dict[str, Any]:
    stale_fields: list[str] = [str(item) for item in dict(observation.quality).get("stale_fields", []) if item]
    warnings: list[str] = [str(item) for item in dict(observation.quality).get("warnings", []) if item]
    missing_symbols = [symbol for symbol in REQUIRED_SYMBOLS if symbol not in observation.symbols]
    missing_history = [
        key
        for key in REQUIRED_HISTORY
        if len(observation.history.get(key, [])) < min_history_points
    ]

    for symbol, payload in observation.symbols.items():
        field_name = f"{symbol}.last"
        if payload.get("last") in {None, ""} and field_name not in stale_fields:
            stale_fields.append(f"{symbol}.last")

    if missing_symbols:
        message = f"missing required symbols: {', '.join(missing_symbols)}"
        if message not in warnings:
            warnings.append(message)
    if missing_history:
        message = f"insufficient history for: {', '.join(missing_history)}"
        if message not in warnings:
            warnings.append(message)
    if stale_fields:
        message = f"stale fields detected: {', '.join(stale_fields)}"
        if message not in warnings:
            warnings.append(message)

    is_complete = not missing_symbols and not missing_history and not stale_fields
    return {
        "is_complete": is_complete,
        "missing_symbols": missing_symbols,
        "missing_history": missing_history,
        "stale_fields": stale_fields,
        "warnings": warnings,
    }
