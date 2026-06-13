"""Backtest feature-store builder for deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import (
    DEFAULT_SECTOR_ETF_SYMBOLS,
    IBKRConnectionConfig,
    IBKRDataPipe,
    IBKRVolRegimeSnapshotRequest,
)
from agentic_vol_regime_app.features.sector_geometry import compute_sector_geometry_metrics


CORE_REGIME_SYMBOLS = ("SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", "VIX9M")


@dataclass(slots=True, frozen=True)
class FeatureStoreBuildResult:
    feature_store_path: str
    rows: int
    start_date: str
    end_date: str
    source_as_of: str
    warnings: list[str]


def _safe_series(values: list[float], size: int) -> pd.Series:
    return pd.Series(list(values)[:size], dtype=float)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=1)
    z = (series - mean) / std.replace({0.0: np.nan})
    return z.replace([np.inf, -np.inf], np.nan)


def _annualized_realized_vol(close_series: pd.Series, window: int) -> pd.Series:
    returns = np.log(close_series / close_series.shift(1))
    return returns.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(252.0) * 100.0


def _rolling_drawdown(close_series: pd.Series, window: int) -> pd.Series:
    rolling_peak = close_series.rolling(window=window, min_periods=window).max()
    drawdown = (rolling_peak - close_series) / rolling_peak.replace({0.0: np.nan})
    return drawdown.clip(lower=0.0)


def _rolling_trend_persistence(close_series: pd.Series, window: int) -> pd.Series:
    up_steps = close_series.diff().ge(0.0).astype(float)
    return up_steps.rolling(window=window, min_periods=window).mean()


def build_feature_store_frame_from_observation(observation: ObservationRecord) -> pd.DataFrame:
    history = {str(key): list(value) for key, value in dict(observation.history).items()}
    required_history_keys = ("SPY_close", "VIX", "VVIX", "VIX9D", "VIX3M")
    missing = [key for key in required_history_keys if not history.get(key)]
    if missing:
        raise RuntimeError(
            "Cannot build backtest feature store because observation history is missing keys: "
            + ", ".join(missing)
        )

    min_len = min(len(history[key]) for key in required_history_keys)
    if min_len < 64:
        raise RuntimeError(
            f"Backtest feature store requires at least 64 history rows; only received {min_len}."
        )

    as_of = pd.to_datetime(observation.as_of).date()
    dates = pd.bdate_range(end=as_of, periods=min_len).date
    frame = pd.DataFrame({"date": dates})
    frame["spy_close"] = _safe_series(history["SPY_close"], min_len).to_numpy()
    frame["vix"] = _safe_series(history["VIX"], min_len).to_numpy()
    frame["vvix"] = _safe_series(history["VVIX"], min_len).to_numpy()
    frame["vix9d"] = _safe_series(history["VIX9D"], min_len).to_numpy()
    frame["vix3m"] = _safe_series(history["VIX3M"], min_len).to_numpy()
    frame["vix6m"] = _safe_series(history.get("VIX6M", history["VIX3M"]), min_len).to_numpy()
    frame["vix9m"] = _safe_series(history.get("VIX9M", history["VIX3M"]), min_len).to_numpy()

    for symbol in DEFAULT_SECTOR_ETF_SYMBOLS:
        key = f"{symbol}_close"
        values = history.get(key)
        if values and len(values) >= min_len:
            frame[key] = _safe_series(values, min_len).to_numpy()
        else:
            frame[key] = np.nan

    frame["spy_return_1d"] = frame["spy_close"].pct_change()
    frame["realized_vol_5d"] = _annualized_realized_vol(frame["spy_close"], window=5)
    frame["realized_vol_21d"] = _annualized_realized_vol(frame["spy_close"], window=21)
    frame["vvix_vix_ratio"] = frame["vvix"] / frame["vix"].replace({0.0: np.nan})
    frame["vix_z_22d"] = _rolling_zscore(frame["vix"], window=22)
    frame["vvix_vix_z_22d"] = _rolling_zscore(frame["vvix_vix_ratio"], window=22)
    frame["vix9d_vix_ratio"] = frame["vix9d"] / frame["vix"].replace({0.0: np.nan})
    frame["vix_vix3m_ratio"] = frame["vix"] / frame["vix3m"].replace({0.0: np.nan})
    frame["term_structure_slope"] = frame["vix3m"] - frame["vix"]
    frame["drawdown_21d"] = _rolling_drawdown(frame["spy_close"], window=21)
    frame["trend_persistence_21d"] = _rolling_trend_persistence(frame["spy_close"], window=21)

    avg_corr: list[float | None] = []
    first_share: list[float | None] = []
    eff_rank: list[float | None] = []
    log_det: list[float | None] = []
    for index in range(min_len):
        partial = {
            key: list(frame[key].iloc[: index + 1].astype(float).values)
            for key in frame.columns
            if key.endswith("_close")
        }
        metrics, _warnings = compute_sector_geometry_metrics(partial, lookback_days=21)
        avg_corr.append(metrics.get("avg_pairwise_corr_21d"))
        first_share.append(metrics.get("first_eigenvalue_share_21d"))
        eff_rank.append(metrics.get("effective_rank_21d"))
        log_det.append(metrics.get("log_det_corr_21d"))
    frame["avg_pairwise_corr_21d"] = avg_corr
    frame["first_eigenvalue_share_21d"] = first_share
    frame["effective_rank_21d"] = eff_rank
    frame["log_det_corr_21d"] = log_det

    required_for_replay = [
        "spy_close",
        "spy_return_1d",
        "vix",
        "vvix",
        "realized_vol_5d",
        "realized_vol_21d",
        "vvix_vix_ratio",
        "vix_z_22d",
        "vvix_vix_z_22d",
        "vix9d_vix_ratio",
        "vix_vix3m_ratio",
        "term_structure_slope",
        "drawdown_21d",
        "trend_persistence_21d",
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required_for_replay).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Feature-store build produced no usable rows after preprocessing.")
    return frame


def _persist_frame(frame: pd.DataFrame, output_path: Path) -> tuple[Path, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        frame.to_parquet(output_path, index=False)
        return output_path, warnings
    except Exception as exc:
        csv_path = output_path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        warnings.append(
            "Failed to write parquet; wrote CSV instead. "
            f"Reason: {exc}"
        )
        return csv_path, warnings


def build_backtest_feature_store_from_ibkr(
    *,
    app_paths: AppPaths | None = None,
    output_path: str | Path | None = None,
    symbol: str = "SPY",
    history_days: int = 1512,
    as_of_date: str | None = None,
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    market_data_type: int = 1,
    exchange: str = "SMART",
    option_exchange: str = "SMART",
    index_exchange: str = "CBOE",
    currency: str = "USD",
    ibkr_data_pipe: IBKRDataPipe | None = None,
) -> FeatureStoreBuildResult:
    paths = app_paths or AppPaths.default()
    resolved_output = Path(output_path) if output_path else paths.root / "data" / "processed" / "features_daily.parquet"
    pipe = ibkr_data_pipe or IBKRDataPipe(
        connection=IBKRConnectionConfig(
            host=host,
            port=int(port),
            client_id=int(client_id),
            readonly=True,
            timeout_seconds=10.0,
            market_data_type=int(market_data_type),
        )
    )
    request_payload: dict[str, Any] = {
        "symbol": symbol.upper().strip() or "SPY",
        "exchange": exchange,
        "option_exchange": option_exchange,
        "index_exchange": index_exchange,
        "currency": currency,
        "history_days": int(history_days),
        "regime_symbols": [symbol.upper().strip() or "SPY", "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", "VIX9M", *DEFAULT_SECTOR_ETF_SYMBOLS],
        "expiry_count": 0,
        "strike_count": 0,
        "min_days_to_expiry": 0,
    }
    if as_of_date:
        request_payload["as_of_date"] = str(as_of_date).strip()

    request = IBKRVolRegimeSnapshotRequest.from_payload(request_payload)
    observation = pipe.fetch_vol_regime_snapshot(request)
    frame = build_feature_store_frame_from_observation(observation)
    persisted_path, warnings = _persist_frame(frame, resolved_output.resolve())
    return FeatureStoreBuildResult(
        feature_store_path=str(persisted_path),
        rows=int(len(frame)),
        start_date=str(frame["date"].min()),
        end_date=str(frame["date"].max()),
        source_as_of=str(observation.as_of),
        warnings=warnings,
    )
