from __future__ import annotations


def leap_exposure_from_brace_score(brace_for_impact_score: float) -> float:
    score = float(brace_for_impact_score)
    if score >= 0.90:
        return 0.00
    if score >= 0.75:
        return 0.25
    if score >= 0.55:
        return 0.50
    if score >= 0.30:
        return 0.75
    return 1.00


def overwrite_action_from_brace_score(brace_for_impact_score: float) -> str:
    score = float(brace_for_impact_score)
    if score < 0.30:
        return "conservative"
    if score < 0.55:
        return "normal"
    return "aggressive"

