"""Volatility regime application built on top of agentic_harness."""

from ._bootstrap import ensure_repo_imports

ensure_repo_imports()

__all__ = [
    "default_agent_path",
    "default_hmm_agent_path",
    "default_hmm_v2_agent_path",
    "default_hmm_v3_agent_path",
    "default_hmm_v3_1_agent_path",
    "default_ibkr_agent_path",
    "default_ml_agent_path",
    "build_backtest_feature_store",
    "load_historical_belief_report",
    "load_latest_live_daily_observation",
    "load_or_run_historical_belief_report",
    "load_recent_hmm_state_history",
    "reset_hmm_persisted_state",
    "run_hmm_replay_backtester",
    "run_live_overwrite_policy_engine",
    "run_overwrite_candidate_scorer",
    "run_policy_backtester",
    "snapshot_hmm_baseline",
    "resume_daily_regime_run",
    "run_daily_regime_agent",
    "run_ibkr_market_data_agent",
    "load_sector_price_history",
    "sync_sector_history",
    "publish_sector_store_to_gcs",
    "verify_sector_store_in_gcs",
    "update_and_publish_sector_history",
    "sync_and_publish_vol_regime_history",
]


def __getattr__(name: str):
    """Keep lightweight publisher imports independent of workflow dependencies."""
    if name not in __all__:
        raise AttributeError(name)
    if name in {"publish_sector_store_to_gcs", "verify_sector_store_in_gcs"}:
        from .data import sector_history_gcs as module
    elif name in {"load_sector_price_history", "sync_sector_history"}:
        from .data import sector_history_store as module
    elif name == "update_and_publish_sector_history":
        from .data import sector_history_update_publish as module
    elif name == "sync_and_publish_vol_regime_history":
        from .data import vol_regime_history_gcs as module
    else:
        from . import app_runtime as module
    return getattr(module, name)
