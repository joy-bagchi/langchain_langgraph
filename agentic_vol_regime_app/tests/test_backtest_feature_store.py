from __future__ import annotations

from datetime import datetime, timezone

from agentic_vol_regime_app.backtest_feature_store import build_feature_store_frame_from_observation
from agentic_vol_regime_app.contracts import ObservationRecord
from agentic_vol_regime_app.data.ibkr_client import DEFAULT_SECTOR_ETF_SYMBOLS


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
    ):
        assert column in frame.columns
