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
    stale_fields: list[str] = []
    warnings: list[str] = []
    missing_symbols = [symbol for symbol in REQUIRED_SYMBOLS if symbol not in observation.symbols]
    missing_history = [
        key
        for key in REQUIRED_HISTORY
        if len(observation.history.get(key, [])) < min_history_points
    ]

    for symbol, payload in observation.symbols.items():
        if payload.get("last") in {None, ""}:
            stale_fields.append(f"{symbol}.last")

    if missing_symbols:
        warnings.append(f"missing required symbols: {', '.join(missing_symbols)}")
    if missing_history:
        warnings.append(f"insufficient history for: {', '.join(missing_history)}")
    if stale_fields:
        warnings.append(f"stale fields detected: {', '.join(stale_fields)}")

    is_complete = not missing_symbols and not missing_history and not stale_fields
    return {
        "is_complete": is_complete,
        "missing_symbols": missing_symbols,
        "missing_history": missing_history,
        "stale_fields": stale_fields,
        "warnings": warnings,
    }
