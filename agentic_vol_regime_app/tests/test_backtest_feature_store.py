from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agentic_vol_regime_app import app_runtime
from agentic_vol_regime_app.backtest_feature_store import (
    STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS,
    _compute_boundary_fetch_windows,
    _is_sector_history_row,
    _is_xlre_late_inception_row,
    _load_cached_history_points,
    _sector_row_meets_relaxed_coverage,
    _upsert_history_points,
    build_feature_store_frame_from_observation,
)
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import DEFAULT_SECTOR_ETF_SYMBOLS
from src.data.yahoo_eod_provider import YahooEODPoint


def test_build_feature_store_frame_from_observation_has_replay_columns() -> None:
    history_days = 300
    history = {
        "SPY_close": [500.0 + (index * 0.4) for index in range(history_days)],
        "VIX": [14.5 + (index * 0.01) for index in range(history_days)],
        "VVIX": [84.0 + (index * 0.04) for index in range(history_days)],
        "VIX9D": [14.0 + (index * 0.008) for index in range(history_days)],
        "VIX3M": [16.0 + (index * 0.009) for index in range(history_days)],
        "VIX6M": [16.8 + (index * 0.009) for index in range(history_days)],
        "VIX9M": [17.4 + (index * 0.009) for index in range(history_days)],
    }
    for symbol in DEFAULT_SECTOR_ETF_SYMBOLS:
        history[f"{symbol}_close"] = [100.0 + (index * 0.2) for index in range(history_days)]
    observation = ObservationRecord(
        schema_version="observation.v1",
        as_of=datetime.now(timezone.utc).isoformat(),
        source="test",
        symbols={
            "SPY": {"last": history["SPY_close"][-1]},
            "VIX": {"last": history["VIX"][-1]},
            "VVIX": {"last": history["VVIX"][-1]},
            "VIX9D": {"last": history["VIX9D"][-1]},
            "VIX3M": {"last": history["VIX3M"][-1]},
        },
        history=history,
        quality={},
        option_chain={},
        provider_metadata={},
    )

    frame = build_feature_store_frame_from_observation(observation)

    assert len(frame) > 200
    for column in (
        "date",
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
        "sector_count_used",
        "xlre_available",
        "geometry_universe_version",
    ):
        assert column in frame.columns


def test_strict_10y_feature_store_builder_rejects_shallow_history() -> None:
    config_path = "agentic_vol_regime_app/configs/backtest/hmm_replay_10y_hmmv4.yaml"

    with pytest.raises(RuntimeError, match="requires at least 2520 trading days"):
        app_runtime.build_backtest_feature_store(
            config_path=config_path,
            history_days=1512,
        )


def test_feature_store_build_allows_missing_optional_vix9d() -> None:
    history_days = 300
    history = {
        "SPY_close": [500.0 + (index * 0.4) for index in range(history_days)],
        "VIX": [14.5 + (index * 0.01) for index in range(history_days)],
        "VVIX": [84.0 + (index * 0.04) for index in range(history_days)],
        "VIX3M": [16.0 + (index * 0.009) for index in range(history_days)],
    }
    for symbol in DEFAULT_SECTOR_ETF_SYMBOLS:
        history[f"{symbol}_close"] = [100.0 + (index * 0.2) for index in range(history_days)]
    observation = ObservationRecord(
        schema_version="observation.v1",
        as_of=datetime.now(timezone.utc).isoformat(),
        source="test",
        symbols={
            "SPY": {"last": history["SPY_close"][-1]},
            "VIX": {"last": history["VIX"][-1]},
            "VVIX": {"last": history["VVIX"][-1]},
            "VIX3M": {"last": history["VIX3M"][-1]},
        },
        history=history,
        quality={},
        option_chain={},
        provider_metadata={},
    )

    frame = build_feature_store_frame_from_observation(observation)
    assert len(frame) > 200
    assert "vix9d_vix_ratio" in frame.columns


def test_feature_store_build_requires_sector_history() -> None:
    history_days = 300
    history = {
        "SPY_close": [500.0 + (index * 0.4) for index in range(history_days)],
        "VIX": [14.5 + (index * 0.01) for index in range(history_days)],
        "VVIX": [84.0 + (index * 0.04) for index in range(history_days)],
        "VIX3M": [16.0 + (index * 0.009) for index in range(history_days)],
    }
    for symbol in DEFAULT_SECTOR_ETF_SYMBOLS:
        if symbol == "XLK":
            continue
        history[f"{symbol}_close"] = [100.0 + (index * 0.2) for index in range(history_days)]
    observation = ObservationRecord(
        schema_version="observation.v1",
        as_of=datetime.now(timezone.utc).isoformat(),
        source="test",
        symbols={
            "SPY": {"last": history["SPY_close"][-1]},
            "VIX": {"last": history["VIX"][-1]},
            "VVIX": {"last": history["VVIX"][-1]},
            "VIX3M": {"last": history["VIX3M"][-1]},
        },
        history=history,
        quality={},
        option_chain={},
        provider_metadata={},
    )

    with pytest.raises(RuntimeError, match="missing keys"):
        build_feature_store_frame_from_observation(observation)


def test_xlre_late_inception_row_is_soft_exempted() -> None:
    row = {
        "symbol": "XLRE",
        "final_earliest_date": "2015-10-08",
        "status": "insufficient",
        "is_optional": False,
    }
    assert _is_xlre_late_inception_row(row, required_start_for_sectors="2013-01-01") is True


def test_non_xlre_late_start_is_not_soft_exempted() -> None:
    row = {
        "symbol": "XLF",
        "final_earliest_date": "2015-10-08",
        "status": "insufficient",
        "is_optional": False,
    }
    assert _is_xlre_late_inception_row(row, required_start_for_sectors="2013-01-01") is False


def test_boundary_fetch_windows_request_only_missing_head_and_tail() -> None:
    windows = _compute_boundary_fetch_windows(
        cached_days=[date(2020, 1, 10), date(2020, 1, 15)],
        requested_start=date(2020, 1, 1),
        requested_end=date(2020, 1, 20),
    )
    assert windows == [
        (date(2020, 1, 1), date(2020, 1, 9)),
        (date(2020, 1, 16), date(2020, 1, 20)),
    ]


def test_upsert_and_load_cached_history_points_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "historical_data.db"
    inserted = _upsert_history_points(
        db_path=db_path,
        symbol="SPY",
        points=[
            YahooEODPoint(day=date(2020, 1, 2), close=320.1, source="yahoo_chart"),
            YahooEODPoint(day=date(2020, 1, 3), close=321.2, source="yahoo_chart"),
        ],
    )
    assert inserted == 2
    loaded = _load_cached_history_points(
        db_path=db_path,
        symbol="SPY",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 5),
    )
    assert len(loaded) == 2
    assert loaded[0].day == date(2020, 1, 2)
    assert loaded[1].close == 321.2


def test_sector_history_row_detection_uses_history_key() -> None:
    assert _is_sector_history_row({"history_key": "XLK_close"}) is True
    assert _is_sector_history_row({"history_key": "SPY_close"}) is False


def test_sector_relaxed_coverage_threshold() -> None:
    assert _sector_row_meets_relaxed_coverage({"row_count": STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}) is True
    assert _sector_row_meets_relaxed_coverage({"row_count": STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS - 1}) is False
