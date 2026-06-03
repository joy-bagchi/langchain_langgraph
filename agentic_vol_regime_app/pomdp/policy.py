"""Policy recommendation heuristics."""

from __future__ import annotations

from agentic_vol_regime_app.contracts import AlertRecord, BeliefRecord, PolicyRecommendationRecord, TransitionProbabilityRecord


def recommend_policy_action(
    belief_record: BeliefRecord,
    transition_record: TransitionProbabilityRecord,
    alert_record: AlertRecord,
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
    return PolicyRecommendationRecord(
        schema_version="policy_recommendation.v1",
        as_of=belief_record.as_of,
        recommended_action=recommended_action,
        confidence=confidence,
        rationale=rationale[:4],
        risk_notes=risk_notes[:4],
        requires_human_review=requires_human_review or recommended_action == "MANUAL_REVIEW",
    )
