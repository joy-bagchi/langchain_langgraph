"""Deterministic feature engineering for regime inference."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from agentic_vol_regime_app.contracts import FeatureRecord, ObservationRecord
from agentic_vol_regime_app.features.sector_geometry import compute_sector_geometry_metrics


def _safe_mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _zscore(value: float, history: list[float]) -> float | None:
    if len(history) < 2:
        return None
    sigma = _sample_std(history)
    if sigma <= 1e-9:
        return 0.0
    return (value - _safe_mean(history)) / sigma


def _annualized_realized_vol(closes: list[float], lookback: int) -> float | None:
    if len(closes) < lookback + 1:
        return None
    window = closes[-(lookback + 1) :]
    returns: list[float] = []
    for index in range(1, len(window)):
        previous = window[index - 1]
        current = window[index]
        if previous <= 0:
            continue
        returns.append(math.log(current / previous))
    if len(returns) < 2:
        return None
    sigma = _sample_std(returns)
    return sigma * math.sqrt(252.0)


def _drawdown(closes: list[float], lookback: int) -> float | None:
    if len(closes) < lookback:
        return None
    window = closes[-lookback:]
    peak = max(window)
    if peak <= 0:
        return None
    return max(0.0, (peak - window[-1]) / peak)


def _trend_persistence(closes: list[float], lookback: int) -> float | None:
    if len(closes) < lookback + 1:
        return None
    window = closes[-(lookback + 1) :]
    positive_steps = 0
    total_steps = 0
    for index in range(1, len(window)):
        total_steps += 1
        if window[index] >= window[index - 1]:
            positive_steps += 1
    if total_steps == 0:
        return None
    return positive_steps / total_steps


def _simple_return(previous: float, current: float) -> float | None:
    if previous <= 0:
        return None
    return (current / previous) - 1.0


def compute_feature_record(
    observation: ObservationRecord,
    *,
    feature_config: dict[str, Any],
) -> FeatureRecord:
    """Compute the deterministic feature set used by the daily report workflow."""
    lookbacks = dict(feature_config.get("lookback_windows", {}))
    zscore_short = int(lookbacks.get("zscore_short", 22))
    rv_short = int(lookbacks.get("rv_short", 5))
    rv_medium = int(lookbacks.get("rv_medium", 21))
    drawdown_window = int(lookbacks.get("drawdown_window", 21))

    spy_closes = list(observation.history.get("SPY_close", []))
    vix_history = list(observation.history.get("VIX", []))
    vvix_history = list(observation.history.get("VVIX", []))
    vix9d_history = list(observation.history.get("VIX9D", []))
    term_structure_symbol = str(
        observation.provider_metadata.get("term_structure_symbol", "VIX3M")
    ).upper()
    term_structure_history = list(
        observation.history.get(term_structure_symbol, observation.history.get("VIX3M", []))
    )

    vix = float(observation.symbols.get("VIX", {}).get("last", 0.0))
    vvix = float(observation.symbols.get("VVIX", {}).get("last", 0.0))
    vix9d = float(observation.symbols.get("VIX9D", {}).get("last", 0.0))
    term_structure_level = float(
        observation.symbols.get(term_structure_symbol, observation.symbols.get("VIX3M", {})).get("last", 0.0)
    )
    spy_last = float(observation.symbols.get("SPY", {}).get("last", 0.0))

    vvix_vix_ratio = vvix / vix if vix else 0.0
    ratio_history = [
        vvix_value / vix_value
        for vvix_value, vix_value in zip(vvix_history[-zscore_short:], vix_history[-zscore_short:])
        if vix_value
    ]
    term_spread_history = [
        back_term_value - vix_value
        for back_term_value, vix_value in zip(term_structure_history[-zscore_short:], vix_history[-zscore_short:])
    ]
    sector_geometry_metrics, _ = compute_sector_geometry_metrics(
        observation.history,
        lookback_days=21,
    )

    rv_5d = _annualized_realized_vol(spy_closes, rv_short)
    rv_21d = _annualized_realized_vol(spy_closes, rv_medium)
    drawdown_21d = _drawdown(spy_closes, drawdown_window)
    trend_persistence_21d = _trend_persistence(spy_closes, drawdown_window)

    missing_features: list[str] = []

    def track(name: str, value: float | str | None) -> float | str | None:
        if value is None:
            missing_features.append(name)
        return value

    term_structure_state = "flat"
    if term_structure_level - vix > 1.0:
        term_structure_state = "contango"
    elif term_structure_level - vix < -1.0:
        term_structure_state = "backwardation"

    features = {
        "spy_last": track("spy_last", spy_last),
        "spy_return_1d": track(
            "spy_return_1d",
            _simple_return(spy_closes[-2], spy_closes[-1]) if len(spy_closes) >= 2 else None,
        ),
        "vix": track("vix", vix),
        "vvix": track("vvix", vvix),
        "vix9d": track("vix9d", vix9d),
        "vix3m": track("vix3m", term_structure_level),
        "term_structure_symbol": term_structure_symbol,
        "vvix_vix_ratio": track("vvix_vix_ratio", vvix_vix_ratio),
        "vvix_vix_z_22d": track("vvix_vix_z_22d", _zscore(vvix_vix_ratio, ratio_history[-zscore_short:])),
        "vix_21d_z": track("vix_21d_z", _zscore(vix, vix_history[-zscore_short:])),
        "vix_z_22d": track("vix_z_22d", _zscore(vix, vix_history[-zscore_short:])),
        "vix9d_vix_ratio": track("vix9d_vix_ratio", vix9d / vix if vix else None),
        "rv_5d": track("rv_5d", rv_5d),
        "rv_21d": track("rv_21d", rv_21d),
        "realized_vol_5d": track("realized_vol_5d", rv_5d * 100.0 if rv_5d is not None else None),
        "realized_vol_21d": track("realized_vol_21d", rv_21d * 100.0 if rv_21d is not None else None),
        "realized_vol_acceleration": track(
            "realized_vol_acceleration",
            rv_5d - rv_21d if rv_5d is not None and rv_21d is not None else None,
        ),
        "vix_rv_spread": track("vix_rv_spread", (vix / 100.0) - rv_21d if rv_21d is not None else None),
        "vix3m_minus_vix": track("vix3m_minus_vix", term_structure_level - vix),
        "vix_vix3m_ratio": track("vix_vix3m_ratio", vix / term_structure_level if term_structure_level else None),
        "term_structure_slope": track("term_structure_slope", term_structure_level - vix),
        "term_structure_flattening": track(
            "term_structure_flattening",
            (term_spread_history[-1] - _safe_mean(term_spread_history[:-1])) if len(term_spread_history) >= 2 else None,
        ),
        "term_structure_state": term_structure_state,
        "trend_persistence_21d": track("trend_persistence_21d", trend_persistence_21d),
        "drawdown_21d": track("drawdown_21d", drawdown_21d),
        "avg_pairwise_corr_21d": track(
            "avg_pairwise_corr_21d",
            sector_geometry_metrics.get("avg_pairwise_corr_21d"),
        ),
        "first_eigenvalue_share_21d": track(
            "first_eigenvalue_share_21d",
            sector_geometry_metrics.get("first_eigenvalue_share_21d"),
        ),
        "effective_rank_21d": track(
            "effective_rank_21d",
            sector_geometry_metrics.get("effective_rank_21d"),
        ),
        "log_det_corr_21d": track(
            "log_det_corr_21d",
            sector_geometry_metrics.get("log_det_corr_21d"),
        ),
    }
    return FeatureRecord(
        schema_version="features.v1",
        as_of=observation.as_of,
        feature_set_version=str(feature_config.get("feature_set_version", "vol_regime_features_v1")),
        features=features,
        missing_features=missing_features,
        lookback_windows={key: int(value) for key, value in lookbacks.items()},
    )
