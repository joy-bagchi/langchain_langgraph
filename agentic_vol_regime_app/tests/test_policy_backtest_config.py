from __future__ import annotations

from src.backtest.policy.policy_backtester import load_policy_backtest_config


def test_load_policy_backtest_config_defaults_from_repo_file() -> None:
    config = load_policy_backtest_config("configs/backtest/policy_backtest.yaml")
    assert config.run_mode in {"tuning", "testing"}
    assert config.train_lookback_days >= 252
    assert config.min_train_rows >= 126
    assert isinstance(config.models, list)
