"""Simple learned belief-state update using a linear regression on SPY realized vol vs VIX."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from agentic_vol_regime_app.contracts import BeliefRecord, FeatureRecord, ObservationRecord
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


def _safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _annualized_realized_vol(window: list[float]) -> float | None:
    if len(window) < 3:
        return None
    returns: list[float] = []
    for index in range(1, len(window)):
        previous = window[index - 1]
        current = window[index]
        if previous <= 0.0 or current <= 0.0:
            continue
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return None
    return _sample_std(returns) * math.sqrt(252.0)


def _rolling_realized_vol_samples(closes: list[float], lookback: int) -> list[float]:
    if len(closes) < lookback + 1:
        return []
    samples: list[float] = []
    for end_index in range(lookback, len(closes)):
        value = _annualized_realized_vol(closes[end_index - lookback : end_index + 1])
        if value is not None:
            samples.append(value * 100.0)
    return samples


def _fit_linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if not xs or not ys:
        return (0.0, 0.0, 0.0)
    if len(xs) != len(ys):
        raise ValueError("Regression inputs must be aligned.")
    if len(xs) == 1:
        return (0.0, ys[0], 0.0)

    x_mean = _safe_mean(xs)
    y_mean = _safe_mean(ys)
    centered_x = [value - x_mean for value in xs]
    centered_y = [value - y_mean for value in ys]
    denominator = sum(value * value for value in centered_x)
    if abs(denominator) <= 1e-9:
        return (0.0, y_mean, 0.0)

    slope = sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y)) / denominator
    intercept = y_mean - (slope * x_mean)
    predictions = [intercept + (slope * value) for value in xs]
    residual_sum = sum((actual - predicted) ** 2 for actual, predicted in zip(ys, predictions))
    total_sum = sum((actual - y_mean) ** 2 for actual in ys)
    r_squared = 0.0 if total_sum <= 1e-9 else 1.0 - (residual_sum / total_sum)
    return (slope, intercept, _clamp(r_squared, 0.0, 1.0))


def update_belief_state_with_linear_regression(
    feature_record: FeatureRecord,
    observation: ObservationRecord,
    *,
    previous_belief: dict[str, float] | None = None,
    lookback: int = 22,
) -> BeliefRecord:
    """Infer the current regime from a learned VIX vs realized-vol relationship."""
    features = dict(feature_record.features)
    spy_closes = list(observation.history.get("SPY_close", []))
    vix_history = [float(value) for value in observation.history.get("VIX", []) if value is not None]
    realized_vol_samples = _rolling_realized_vol_samples(spy_closes, lookback)
    sample_count = min(len(realized_vol_samples), len(vix_history))

    if sample_count >= 2:
        xs = realized_vol_samples[-sample_count:]
        ys = vix_history[-sample_count:]
        slope, intercept, r_squared = _fit_linear_regression(xs, ys)
        if slope <= 0.0:
            x_mean = _safe_mean(xs)
            y_mean = _safe_mean(ys)
            slope = max(y_mean / max(x_mean, 1.0), 0.5)
            intercept = max(0.0, y_mean - (slope * x_mean))
            r_squared = min(r_squared, 0.2)
        rmse = math.sqrt(
            _safe_mean(
                [(actual - (intercept + (slope * value))) ** 2 for value, actual in zip(xs, ys)]
            )
        )
    else:
        current_rv = _feature(features, "rv_21d") * 100.0
        current_vix = _feature(features, "vix")
        slope = 1.0
        intercept = 0.0
        r_squared = 0.0
        rmse = 1.0
        xs = [current_rv]
        ys = [current_vix]

    current_rv = _feature(features, "rv_21d") * 100.0
    current_vix = _feature(features, "vix")
    predicted_vix = intercept + (slope * current_rv)
    residual = current_vix - predicted_vix
    normalized_gap = residual / max(rmse, 1.0)

    vvix_ratio = _feature(features, "vvix_vix_ratio")
    vvix_ratio_z = _feature(features, "vvix_vix_z_22d")
    drawdown = _feature(features, "drawdown_21d")
    term_flattening = _feature(features, "term_structure_flattening")
    trend_persistence = _feature(features, "trend_persistence_21d", 0.5)
    term_state = str(features.get("term_structure_state", "flat"))

    scores = {
        "STABLE_LOW_VOL_TREND": 1.2,
        "MID_VOL_CHOP": 1.0,
        "VOL_EXPANSION_TRANSITION": 0.9,
        "HIGH_VOL_RISK_OFF": 0.7,
        "PANIC_CONVEXITY_STRESS": 0.35,
        "POST_PANIC_COMPRESSION": 0.55,
    }
    drivers: list[str] = []

    if sample_count >= 2:
        drivers.append(
            f"Linear fit used {sample_count} realized-vol samples with slope {slope:.2f} and R^2 {r_squared:.2f}."
        )
    else:
        drivers.append("Historical sample size was thin, so the linear model ran in low-confidence fallback mode.")

    drivers.append(
        f"Current 22d realized vol is {current_rv:.1f} versus model-implied VIX {predicted_vix:.1f} and observed VIX {current_vix:.1f}."
    )

    if current_vix <= 17.5:
        scores["STABLE_LOW_VOL_TREND"] += 0.9
    if term_state == "contango":
        scores["STABLE_LOW_VOL_TREND"] += 0.7
    if current_vix <= 19.0 and abs(normalized_gap) <= 0.6 and term_state == "contango":
        scores["STABLE_LOW_VOL_TREND"] += 0.85
    if normalized_gap <= -0.75:
        scores["STABLE_LOW_VOL_TREND"] += 0.7
        scores["POST_PANIC_COMPRESSION"] += 0.8
        drivers.append("Observed VIX is below the regression-implied level, which points to compression.")

    if 18.5 <= current_vix < 22.5:
        scores["MID_VOL_CHOP"] += 0.75
    if abs(normalized_gap) <= 0.8:
        scores["MID_VOL_CHOP"] += 0.45

    if normalized_gap >= 0.75:
        scores["VOL_EXPANSION_TRANSITION"] += 1.0
        drivers.append("Observed VIX is materially above the regression-implied level, flagging expansion risk.")
    if vvix_ratio_z >= 0.7:
        scores["VOL_EXPANSION_TRANSITION"] += 0.6
    if term_flattening < -0.2:
        scores["VOL_EXPANSION_TRANSITION"] += 0.5

    if normalized_gap >= 1.4 or current_vix >= 24.0:
        scores["HIGH_VOL_RISK_OFF"] += 1.0
    if term_state == "backwardation":
        scores["HIGH_VOL_RISK_OFF"] += 0.85
    if drawdown >= 0.035:
        scores["HIGH_VOL_RISK_OFF"] += 0.7

    if normalized_gap >= 2.25 and vvix_ratio >= 6.4:
        scores["PANIC_CONVEXITY_STRESS"] += 1.35
        drivers.append("VIX and convexity are both far above the fitted relationship.")
    if drawdown >= 0.07:
        scores["PANIC_CONVEXITY_STRESS"] += 0.8

    if normalized_gap <= -0.8 and current_vix > 20.0 and term_state == "contango":
        scores["POST_PANIC_COMPRESSION"] += 1.0
    if trend_persistence >= 0.6 and normalized_gap <= 0.0:
        scores["POST_PANIC_COMPRESSION"] += 0.35

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
    fit_bonus = 0.25 * r_squared
    confidence = round(
        _clamp(0.42 + fit_bonus + (max(beliefs.values()) * 0.38) - (normalized_entropy * 0.2), 0.0, 1.0),
        6,
    )

    return BeliefRecord(
        schema_version="belief.v1",
        as_of=feature_record.as_of,
        model_version="linear_regression_regime_v1",
        beliefs={regime: round(beliefs.get(regime, 0.0), 6) for regime in REGIMES},
        belief_delta=belief_delta,
        entropy=round(normalized_entropy, 6),
        confidence=confidence,
        drivers=drivers[:5],
    )
