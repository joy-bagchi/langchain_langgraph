"""Deterministic near-term transition probability heuristics."""

from __future__ import annotations

from typing import Any

from agentic_vol_regime_app.contracts import BeliefRecord, FeatureRecord, TransitionProbabilityRecord


def _feature(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.01, upper: float = 0.95) -> float:
    return max(lower, min(upper, value))


def _confirming_features(feature_record: FeatureRecord) -> tuple[int, list[str]]:
    features = dict(feature_record.features)
    confirmations: list[str] = []
    if _feature(features, "vvix_vix_z_22d") > 1.0:
        confirmations.append("VVIX/VIX z-score")
    if _feature(features, "realized_vol_acceleration") > 0.03:
        confirmations.append("realized volatility acceleration")
    if _feature(features, "term_structure_flattening") < -0.4:
        confirmations.append("term structure flattening")
    if _feature(features, "drawdown_21d") > 0.04:
        confirmations.append("SPY drawdown pressure")
    if _feature(features, "vix_rv_spread") > 0.04:
        confirmations.append("IV over realized-vol spread")
    return len(confirmations), confirmations


def estimate_transition_probabilities(
    feature_record: FeatureRecord,
    belief_record: BeliefRecord,
) -> TransitionProbabilityRecord:
    """Estimate simple short-horizon transition probabilities."""
    features = dict(feature_record.features)
    beliefs = dict(belief_record.beliefs)

    expansion = beliefs.get("VOL_EXPANSION_TRANSITION", 0.0)
    high_vol = beliefs.get("HIGH_VOL_RISK_OFF", 0.0)
    chop = beliefs.get("MID_VOL_CHOP", 0.0)

    vvix_ratio_z = _feature(features, "vvix_vix_z_22d")
    rv_acceleration = _feature(features, "realized_vol_acceleration")
    vix_z = _feature(features, "vix_21d_z")
    drawdown = _feature(features, "drawdown_21d")
    term_flattening = _feature(features, "term_structure_flattening")
    term_spread = _feature(features, "vix3m_minus_vix")

    confirming_count, confirming_factors = _confirming_features(feature_record)
    confirming_boost = min(confirming_count / 4.0, 1.0)

    transition_probabilities = {
        "vol_expansion_5d": _clamp(
            0.05
            + (0.45 * expansion)
            + (0.10 * max(vvix_ratio_z, 0.0))
            + (0.10 * max(rv_acceleration, 0.0))
            + (0.08 * max(-term_flattening, 0.0))
            + (0.10 * confirming_boost)
        ),
        "vix_spike_5d": _clamp(
            0.04
            + (0.25 * expansion)
            + (0.20 * high_vol)
            + (0.08 * max(vix_z, 0.0))
            + (0.10 * confirming_boost)
        ),
        "vix_spike_10d": _clamp(
            0.06
            + (0.20 * expansion)
            + (0.26 * high_vol)
            + (0.08 * max(vix_z, 0.0))
            + (0.12 * confirming_boost)
        ),
        "vix_explosion_10d": _clamp(
            0.02
            + (0.58 * high_vol)
            + (0.12 * confirming_boost)
            + (0.08 * max(drawdown, 0.0))
        ),
        "risk_off_transition_10d": _clamp(
            0.05
            + (0.22 * expansion)
            + (0.25 * high_vol)
            + (0.10 * max(drawdown, 0.0) * 10.0)
            + (0.08 * confirming_boost)
        ),
        "vol_compression_10d": _clamp(
            0.04
            + (0.30 * chop)
            + (0.07 * max(term_spread, 0.0))
            - (0.18 * high_vol)
        ),
    }

    return TransitionProbabilityRecord(
        schema_version="transition_probabilities.v1",
        as_of=feature_record.as_of,
        model_version="transition_model_v1",
        transition_probabilities={key: round(value, 6) for key, value in transition_probabilities.items()},
        top_predictive_factors=confirming_factors[:4],
        confirming_features_count=confirming_count,
    )
