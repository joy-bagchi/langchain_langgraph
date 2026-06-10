"""Policy recommendation heuristics."""

from __future__ import annotations

import math

from agentic_vol_regime_app.contracts import (
    AlertRecord,
    BeliefRecord,
    FeatureRecord,
    PolicyRecommendationRecord,
    TransitionProbabilityRecord,
)


def _strike_increment(spot: float) -> float:
    if spot >= 1000:
        return 5.0
    if spot >= 200:
        return 1.0
    return 0.5


def _round_up_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return value
    return math.ceil(value / increment) * increment


def _overwrite_contract(
    *,
    action: str,
    spot: float | None,
    hmm_record: dict[str, object] | None = None,
) -> tuple[float | None, int | None, str | None]:
    if spot is None or spot <= 0:
        return (None, None, None)
    if action in {"NO_OVERWRITE", "MANUAL_REVIEW"}:
        return (None, None, None)

    increment = _strike_increment(spot)
    offsets = {
        "LIGHT_OVERWRITE": (0.005, 1, "Light overwrite keeps the call slightly out of the money."),
        "MEDIUM_OVERWRITE": (0.008, 1, "Medium overwrite moves the call further out to balance premium and upside room."),
        "AGGRESSIVE_OVERWRITE": (0.012, 1, "Aggressive overwrite caps closer upside risk with a tighter short-dated call."),
        "REDUCE_OVERWRITE": (0.015, 2, "Reduced overwrite stance widens strike distance and extends DTE to preserve upside."),
    }
    pct_otm, dte, note = offsets.get(
        action,
        (0.006, 1, "Overwrite strike selected from the current spot and regime posture."),
    )
    if hmm_record and float(hmm_record.get("current_state_expected_duration_days", 0.0) or 0.0) > 0.0:
        expected_duration = float(hmm_record.get("current_state_expected_duration_days", 0.0) or 0.0)
        transition_probabilities = dict(hmm_record.get("transition_probabilities", {}) or {})
        elevated_transition_risk = max(
            float(transition_probabilities.get("to_high_vol_stress_5d", 0.0) or 0.0),
            float(transition_probabilities.get("to_vol_expansion_or_high_vol_5d", 0.0) or 0.0),
        )
        if expected_duration <= 3.0:
            dte = 1
        elif expected_duration <= 7.0:
            dte = 5
        elif expected_duration <= 14.0:
            dte = 10
        else:
            dte = 17
        if elevated_transition_risk >= 0.35:
            dte = min(dte, 3)
            note = "HMM transition risk is elevated, so the overwrite duration was shortened."
        else:
            note = f"HMM expected regime duration suggests approximately {dte} DTE."
    strike = _round_up_to_increment(spot * (1.0 + pct_otm), increment)
    return (float(strike), int(dte), note)


def recommend_policy_action(
    feature_record: FeatureRecord,
    belief_record: BeliefRecord,
    transition_record: TransitionProbabilityRecord,
    alert_record: AlertRecord,
    *,
    hmm_record: dict[str, object] | None = None,
) -> PolicyRecommendationRecord:
    """Map the current belief state to a recommended overwrite posture."""
    beliefs = dict(belief_record.beliefs)
    transitions = dict(transition_record.transition_probabilities)
    severity = alert_record.severity

    stable = beliefs.get("STABLE_LOW_VOL_TREND", 0.0)
    expansion = beliefs.get("VOL_EXPANSION_TRANSITION", 0.0)
    high_vol = beliefs.get("HIGH_VOL_RISK_OFF", 0.0)
    panic = beliefs.get("PANIC_CONVEXITY_STRESS", 0.0)
    post_panic = beliefs.get("POST_PANIC_COMPRESSION", 0.0)

    recommended_action = "LIGHT_OVERWRITE"
    rationale: list[str] = []
    risk_notes: list[str] = []
    requires_human_review = severity in {"HIGH_RISK", "CRITICAL"}

    if panic >= 0.22 or severity == "CRITICAL":
        recommended_action = "MANUAL_REVIEW"
        rationale.append("Convexity stress or critical alert conditions are active.")
        risk_notes.append("Do not add fresh overwrite exposure without review.")
    elif high_vol >= 0.28 or transitions.get("risk_off_transition_10d", 0.0) >= 0.40:
        recommended_action = "AGGRESSIVE_OVERWRITE"
        rationale.append("High-volatility risk-off regime is materially represented.")
        risk_notes.append("Monitor rebound risk before capping too tightly.")
    elif expansion >= 0.24 or transitions.get("vol_expansion_5d", 0.0) >= 0.35:
        recommended_action = "MEDIUM_OVERWRITE"
        rationale.append("Transition risk is elevated and argues for additional premium capture.")
    elif post_panic >= 0.24:
        recommended_action = "REDUCE_OVERWRITE"
        rationale.append("Post-panic compression favors preserving upside rebound room.")
        risk_notes.append("Avoid replacing downside fear with fresh upside truncation.")
    elif stable >= 0.42 and severity in {"NONE", "WATCH"}:
        recommended_action = "NO_OVERWRITE"
        rationale.append("Stable low-volatility trend remains the dominant regime.")
        risk_notes.append("Tight overwrites may truncate upside more than they help.")
    else:
        rationale.append("Mixed signals argue for a moderate premium stance.")

    if severity == "WARNING" and recommended_action == "NO_OVERWRITE":
        recommended_action = "LIGHT_OVERWRITE"
        rationale.append("Warning-level transition risk nudges posture away from zero overwrite.")

    confidence = round(min(0.95, max(belief_record.confidence, max(beliefs.values()) + 0.1)), 6)
    spot = feature_record.features.get("spy_last")
    overwrite_call_strike, overwrite_dte, overwrite_rationale = _overwrite_contract(
        action=recommended_action,
        spot=float(spot) if spot is not None else None,
        hmm_record=hmm_record,
    )
    return PolicyRecommendationRecord(
        schema_version="policy_recommendation.v1",
        as_of=belief_record.as_of,
        recommended_action=recommended_action,
        confidence=confidence,
        rationale=rationale[:4],
        risk_notes=risk_notes[:4],
        requires_human_review=requires_human_review or recommended_action == "MANUAL_REVIEW",
        overwrite_call_strike=overwrite_call_strike,
        overwrite_dte=overwrite_dte,
        overwrite_rationale=overwrite_rationale,
    )
