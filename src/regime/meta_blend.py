from __future__ import annotations

from dataclasses import dataclass


STATE_ORDER = (
    "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF",
)

STATE_SEVERITY = {
    "STABLE_LOW_VOL_TREND": 0,
    "MID_VOL_CHOP": 1,
    "VOL_EXPANSION_TRANSITION": 2,
    "HIGH_VOL_RISK_OFF": 3,
}

STATE_RISK_SCORE = {
    "STABLE_LOW_VOL_TREND": 0.00,
    "MID_VOL_CHOP": 0.33,
    "VOL_EXPANSION_TRANSITION": 0.67,
    "HIGH_VOL_RISK_OFF": 1.00,
}


def score_to_state(score: float) -> str:
    value = float(min(1.0, max(0.0, score)))
    if value < 0.25:
        return "STABLE_LOW_VOL_TREND"
    if value < 0.50:
        return "MID_VOL_CHOP"
    if value < 0.75:
        return "VOL_EXPANSION_TRANSITION"
    return "HIGH_VOL_RISK_OFF"


def core_vol_risk_score(core_state_probabilities: dict[str, float]) -> float:
    risk = 0.0
    for state, probability in core_state_probabilities.items():
        risk += float(probability) * float(STATE_RISK_SCORE.get(state, 0.0))
    return float(min(1.0, max(0.0, risk)))


@dataclass(slots=True, frozen=True)
class MetaBlendResult:
    final_risk_score: float
    final_regime: str
    confidence_adjustment: str
    downgrade_levels: int
    downgrade_cap_applied: bool
    rationale: list[str]


def blend_with_geometry_modifier(
    *,
    core_vol_state: str,
    core_vol_confidence: float,
    core_vol_risk_score_value: float,
    geometry_stress_score: float,
    core_vol_weight: float = 0.75,
    geometry_weight: float = 0.25,
) -> MetaBlendResult:
    total_weight = float(core_vol_weight) + float(geometry_weight)
    if total_weight <= 0.0:
        core_weight = 0.75
        geo_weight = 0.25
    else:
        core_weight = float(core_vol_weight) / total_weight
        geo_weight = float(geometry_weight) / total_weight
    raw_score = (core_weight * float(core_vol_risk_score_value)) + (geo_weight * float(geometry_stress_score))
    final_score = float(min(1.0, max(0.0, raw_score)))
    raw_state = score_to_state(final_score)
    core_state = str(core_vol_state or "STABLE_LOW_VOL_TREND")
    core_sev = int(STATE_SEVERITY.get(core_state, 0))
    final_sev = int(STATE_SEVERITY.get(raw_state, 0))
    cap_applied = False

    allow_deep_downgrade = bool(float(geometry_stress_score) < 0.20 and float(core_vol_confidence) < 0.55)
    if core_state in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"} and not allow_deep_downgrade:
        minimum_allowed = max(0, core_sev - 1)
        if final_sev < minimum_allowed:
            final_sev = minimum_allowed
            cap_applied = True

    final_regime = STATE_ORDER[final_sev]
    downgrade_levels = int(max(0, core_sev - final_sev))
    confidence_adjustment = (
        "increase"
        if (final_sev > core_sev or geometry_stress_score >= 0.55)
        else "decrease"
        if (final_sev < core_sev and geometry_stress_score < 0.45)
        else "neutral"
    )
    rationale = [
        f"Final risk score uses weighted blend: core={core_weight:.2f}, geometry={geo_weight:.2f}.",
        f"Core state={core_state} (confidence={float(core_vol_confidence):.2f}), geometry_stress={float(geometry_stress_score):.2f}.",
    ]
    if cap_applied:
        rationale.append("Geometry does not confirm stress, but downgrade was capped to one severity level.")
    return MetaBlendResult(
        final_risk_score=float(round(final_score, 6)),
        final_regime=final_regime,
        confidence_adjustment=confidence_adjustment,
        downgrade_levels=downgrade_levels,
        downgrade_cap_applied=cap_applied,
        rationale=rationale,
    )
