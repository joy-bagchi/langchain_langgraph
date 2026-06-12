"""Sector-correlation and geometry helpers for HMM feature variants."""

from __future__ import annotations

import math
from statistics import fmean

import numpy as np


SECTOR_ETF_UNIVERSE = (
    "XLK",
    "XLF",
    "XLE",
    "XLY",
    "XLP",
    "XLI",
    "XLB",
    "XLV",
    "XLU",
    "XLRE",
)


def _daily_returns(window: list[float]) -> list[float] | None:
    if len(window) < 2:
        return None
    returns: list[float] = []
    for index in range(1, len(window)):
        previous = float(window[index - 1])
        current = float(window[index])
        if previous <= 0 or current <= 0:
            return None
        returns.append((current / previous) - 1.0)
    return returns if len(returns) >= 2 else None


def build_sector_return_matrix(
    history: dict[str, list[float]],
    *,
    lookback_days: int,
    sector_symbols: tuple[str, ...] = SECTOR_ETF_UNIVERSE,
) -> tuple[np.ndarray | None, list[str]]:
    """Build aligned trailing sector-return rows from close history."""
    if lookback_days < 2:
        return None, ["lookback_days must be at least 2 for sector-correlation features."]

    series_list: list[list[float]] = []
    missing_symbols: list[str] = []
    for symbol in sector_symbols:
        raw_values = [float(value) for value in history.get(f"{symbol}_close", []) if value is not None]
        if len(raw_values) < lookback_days:
            missing_symbols.append(symbol)
            continue
        returns = _daily_returns(raw_values[-lookback_days:])
        if returns is None:
            missing_symbols.append(symbol)
            continue
        series_list.append(returns)

    if missing_symbols:
        return None, [f"Missing sector history for: {', '.join(missing_symbols)}"]
    if len(series_list) < 2:
        return None, ["Need at least two sector return series for sector-correlation features."]
    return np.asarray(series_list, dtype=float), []


def compute_sector_geometry_metrics(
    history: dict[str, list[float]],
    *,
    lookback_days: int = 21,
    epsilon: float = 1e-6,
    sector_symbols: tuple[str, ...] = SECTOR_ETF_UNIVERSE,
) -> tuple[dict[str, float], list[str]]:
    """Compute compressed sector-correlation metrics for HMM v2/v3 style models."""
    return_matrix, warnings = build_sector_return_matrix(
        history,
        lookback_days=lookback_days,
        sector_symbols=sector_symbols,
    )
    if return_matrix is None:
        return {}, warnings

    correlation = np.corrcoef(return_matrix)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        return {}, ["Sector correlation matrix could not be constructed."]

    diag_indices = np.diag_indices_from(correlation)
    correlation = np.asarray(correlation, dtype=float)
    correlation[diag_indices] = 1.0
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    avg_pairwise_corr = float(fmean(float(value) for value in off_diagonal)) if off_diagonal.size else 0.0

    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    eigen_sum = float(np.sum(eigenvalues))
    if eigen_sum <= 1e-12:
        return {}, ["Sector correlation eigenvalues were degenerate."]

    shares = eigenvalues / eigen_sum
    largest_share = float(np.max(shares))
    stable_shares = np.clip(shares, 1e-12, 1.0)
    effective_rank = float(math.exp(-float(np.sum(stable_shares * np.log(stable_shares)))))

    regularized = correlation + (float(epsilon) * np.eye(correlation.shape[0]))
    sign, log_det = np.linalg.slogdet(regularized)
    if sign <= 0:
        return {}, ["Regularized sector correlation matrix was not positive definite."]

    return {
        "avg_pairwise_corr_21d": round(avg_pairwise_corr, 6),
        "first_eigenvalue_share_21d": round(largest_share, 6),
        "effective_rank_21d": round(effective_rank, 6),
        "log_det_corr_21d": round(float(log_det), 6),
    }, []
