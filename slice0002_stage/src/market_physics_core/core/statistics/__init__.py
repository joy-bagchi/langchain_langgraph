"""Pure descriptive statistics."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + weight * (
        sorted_values[upper] - sorted_values[lower]
    )


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sample_standard_deviation": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "median": statistics.median(cleaned),
        "sample_standard_deviation": (
            statistics.stdev(cleaned) if len(cleaned) >= 2 else None
        ),
        "q1": _quantile(cleaned, 0.25),
        "q3": _quantile(cleaned, 0.75),
        "minimum": cleaned[0],
        "maximum": cleaned[-1],
    }


__all__ = ["describe"]
