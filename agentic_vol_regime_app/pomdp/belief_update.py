"""Heuristic belief-state update."""

from __future__ import annotations

import math
from typing import Any

from agentic_vol_regime_app.contracts import BeliefRecord, FeatureRecord
from agentic_vol_regime_app.pomdp.states import REGIMES


def _feature(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    anchor = max(scores.values())
    numerators = {key: math.exp(value - anchor) for key, value in scores.items()}
    total = sum(numerators.values()) or 1.0
    return {key: value / total for key, value in numerators.items()}


def _entropy(probabilities: dict[str, float]) -> float:
    return -sum(prob * math.log(prob) for prob in probabilities.values() if prob > 0.0)


def update_belief_state(
    feature_record: FeatureRecord,
    *,
    previous_belief: dict[str, float] | None = None,
) -> BeliefRecord:
    """Compute a deterministic hidden-state belief vector from current features."""
    features = dict(feature_record.features)
    vix = _feature(features, "vix")
    vvix_ratio_z = _feature(features, "vvix_vix_z_22d")
    vix_z = _feature(features, "vix_21d_z")
    rv_acceleration = _feature(features, "realized_vol_acceleration")
    term_flattening = _feature(features, "term_structure_flattening")
    drawdown = _feature(features, "drawdown_21d")
    trend_persistence = _feature(features, "trend_persistence_21d", 0.5)
    vix_spread = _feature(features, "vix_rv_spread")
    term_spread = _feature(features, "vix3m_minus_vix")
    term_state = str(features.get("term_structure_state", "flat"))

    scores = {
        "STABLE_LOW_VOL_TREND": 1.8,
        "MID_VOL_CHOP": 1.3,
        "VOL_EXPANSION_TRANSITION": 1.0,
        "HIGH_VOL_RISK_OFF": 0.8,
    }
    drivers: list[str] = []

    if vix <= 18.5:
        scores["STABLE_LOW_VOL_TREND"] += 1.0
        drivers.append("VIX remains anchored in a low-volatility range.")
    if trend_persistence >= 0.65:
        scores["STABLE_LOW_VOL_TREND"] += 0.9
        drivers.append("Trend persistence remains supportive of a stable trend.")
    if term_state == "contango":
        scores["STABLE_LOW_VOL_TREND"] += 0.8
    if vvix_ratio_z > 0.8:
        scores["VOL_EXPANSION_TRANSITION"] += 1.1
        drivers.append("VVIX/VIX z-score is elevated versus recent history.")
    if rv_acceleration > 0.03:
        scores["VOL_EXPANSION_TRANSITION"] += 0.8
        drivers.append("Realized volatility acceleration turned positive.")
    if term_flattening < -0.4:
        scores["VOL_EXPANSION_TRANSITION"] += 0.9
        drivers.append("Term structure is flattening versus its recent average.")
    if 18.5 < vix < 24.0:
        scores["MID_VOL_CHOP"] += 0.8
    if 0.4 <= trend_persistence <= 0.6:
        scores["MID_VOL_CHOP"] += 0.6
    if vix >= 24.0 or drawdown >= 0.045:
        scores["HIGH_VOL_RISK_OFF"] += 1.2
        drivers.append("Risk-off features are active via elevated VIX or drawdown.")
    if term_state == "backwardation":
        scores["HIGH_VOL_RISK_OFF"] += 1.0
    if vix >= 32.0 and vvix_ratio_z >= 1.4:
        scores["HIGH_VOL_RISK_OFF"] += 1.8
        drivers.append("Convexity stress signals are simultaneously elevated.")
    if drawdown >= 0.08:
        scores["HIGH_VOL_RISK_OFF"] += 1.1
    if vix >= 24.0 and term_spread > 1.5 and trend_persistence > 0.5:
        scores["MID_VOL_CHOP"] += 1.0
        scores["VOL_EXPANSION_TRANSITION"] -= 0.2
        drivers.append("Volatility remains elevated but the curve has re-steepened into a choppier regime.")
    if vix_spread > 0.05:
        scores["VOL_EXPANSION_TRANSITION"] += 0.4
        scores["HIGH_VOL_RISK_OFF"] += 0.3

    beliefs = _softmax(scores)
    if previous_belief:
        smoothed: dict[str, float] = {}
        for regime in REGIMES:
            prior = float(previous_belief.get(regime, 1.0 / len(REGIMES)))
            smoothed[regime] = (0.65 * beliefs.get(regime, 0.0)) + (0.35 * prior)
        total = sum(smoothed.values()) or 1.0
        beliefs = {regime: value / total for regime, value in smoothed.items()}

    base_previous = previous_belief or {regime: 1.0 / len(REGIMES) for regime in REGIMES}
    belief_delta = {
        regime: round(beliefs.get(regime, 0.0) - float(base_previous.get(regime, 0.0)), 6)
        for regime in REGIMES
    }
    entropy = _entropy(beliefs)
    normalized_entropy = _clamp(entropy / math.log(len(REGIMES)), 0.0, 1.0)
    confidence = round(1.0 - (normalized_entropy * 0.55) + (max(beliefs.values()) * 0.45), 6)

    return BeliefRecord(
        schema_version="belief.v1",
        as_of=feature_record.as_of,
        model_version="belief_model_v1",
        beliefs={regime: round(beliefs.get(regime, 0.0), 6) for regime in REGIMES},
        belief_delta=belief_delta,
        entropy=round(normalized_entropy, 6),
        confidence=round(_clamp(confidence, 0.0, 1.0), 6),
        drivers=drivers[:5],
    )
