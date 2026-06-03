"""Deterministic predictive alert generation."""

from __future__ import annotations

from uuid import uuid4

from agentic_vol_regime_app.contracts import AlertRecord, BeliefRecord, FeatureRecord, TransitionProbabilityRecord


SEVERITY_ORDER = ("NONE", "WATCH", "WARNING", "HIGH_RISK", "CRITICAL")


def _meets_thresholds(values: dict[str, float], thresholds: dict[str, float]) -> bool:
    for key, threshold in thresholds.items():
        if key == "confirming_features_min":
            continue
        if values.get(key, 0.0) < float(threshold):
            return False
    return True


def build_alert_record(
    feature_record: FeatureRecord,
    belief_record: BeliefRecord,
    transition_record: TransitionProbabilityRecord,
    *,
    thresholds: dict[str, dict[str, float]],
) -> AlertRecord:
    """Generate a predictive alert based on belief and transition risk."""
    beliefs = dict(belief_record.beliefs)
    probabilities = dict(transition_record.transition_probabilities)
    features = dict(feature_record.features)

    threshold_inputs = {
        **probabilities,
        "transition_belief": beliefs.get("VOL_EXPANSION_TRANSITION", 0.0),
        "high_vol_belief": beliefs.get("HIGH_VOL_RISK_OFF", 0.0),
        "panic_belief": beliefs.get("PANIC_CONVEXITY_STRESS", 0.0),
    }
    confirming_features_count = transition_record.confirming_features_count
    drivers = list(transition_record.top_predictive_factors)
    if features.get("term_structure_state") == "backwardation":
        drivers.append("term structure entered backwardation")

    severity = "NONE"
    for candidate in ("critical", "high_risk", "warning", "watch"):
        candidate_thresholds = dict(thresholds.get(candidate, {}))
        minimum_confirmations = int(candidate_thresholds.get("confirming_features_min", 0) or 0)
        if minimum_confirmations and confirming_features_count < minimum_confirmations:
            continue
        if _meets_thresholds(threshold_inputs, candidate_thresholds):
            severity = candidate.upper()
            break

    headline_map = {
        "NONE": "No elevated predictive volatility alert",
        "WATCH": "Transition probability has risen but confirmation remains limited",
        "WARNING": "Volatility expansion risk is rising with confirming signals",
        "HIGH_RISK": "Probability of a near-term volatility spike is elevated",
        "CRITICAL": "Convexity stress risk is extreme and requires manual review",
    }
    review_map = {
        "NONE": [],
        "WATCH": ["Continue monitoring signal persistence before adjusting posture."],
        "WARNING": ["Review overwrite sizing before opening new tight caps."],
        "HIGH_RISK": [
            "Review existing overwrite exposure.",
            "Avoid initiating tight new short calls until risk stabilizes.",
        ],
        "CRITICAL": [
            "Escalate for human review before adding or rolling overwrite exposure.",
            "Check short-call delta risk and downside convexity exposure.",
        ],
    }

    return AlertRecord(
        schema_version="alert.v1",
        alert_id=str(uuid4()),
        as_of=feature_record.as_of,
        severity=severity,
        alert_type="PREDICTIVE_VOL_EXPANSION",
        headline=headline_map[severity],
        probabilities={
            "vol_expansion_5d": round(probabilities.get("vol_expansion_5d", 0.0), 6),
            "vix_spike_10d": round(probabilities.get("vix_spike_10d", 0.0), 6),
            "vix_explosion_10d": round(probabilities.get("vix_explosion_10d", 0.0), 6),
        },
        belief_state={
            "VOL_EXPANSION_TRANSITION": round(beliefs.get("VOL_EXPANSION_TRANSITION", 0.0), 6),
            "HIGH_VOL_RISK_OFF": round(beliefs.get("HIGH_VOL_RISK_OFF", 0.0), 6),
            "PANIC_CONVEXITY_STRESS": round(beliefs.get("PANIC_CONVEXITY_STRESS", 0.0), 6),
        },
        drivers=drivers[:4],
        recommended_review=review_map[severity],
        requires_human_review=severity in {"HIGH_RISK", "CRITICAL"},
    )
