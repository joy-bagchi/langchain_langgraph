from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.hmm_replay.replay_config import load_replay_config
from src.backtest.hmm_replay.replay_runner import run_hmm_replay


def _strict_replay_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=40)
    return pd.DataFrame(
        {
            "date": dates.date,
            "spy_close": [300.0 + idx for idx in range(len(dates))],
            "vix": [15.0 + (idx * 0.1) for idx in range(len(dates))],
            "vvix": [90.0 + (idx * 0.2) for idx in range(len(dates))],
            "vix9d_vix_ratio": [0.95 for _ in range(len(dates))],
            "vix_vix3m_ratio": [0.98 for _ in range(len(dates))],
            "realized_vol_5d": [12.0 for _ in range(len(dates))],
            "realized_vol_21d": [14.0 for _ in range(len(dates))],
            "drawdown_21d": [0.02 for _ in range(len(dates))],
            "trend_persistence_21d": [0.6 for _ in range(len(dates))],
            "avg_pairwise_corr_21d": [0.4 for _ in range(len(dates))],
            "first_eigenvalue_share_21d": [0.5 for _ in range(len(dates))],
            "effective_rank_21d": [4.0 for _ in range(len(dates))],
            "log_det_corr_21d": [-1.2 for _ in range(len(dates))],
            "regime_target": ["Stable Low-Vol" for _ in range(len(dates))],
        }
    )


def test_hmmv4_10y_config_is_strict() -> None:
    config = load_replay_config("agentic_vol_regime_app/configs/backtest/hmm_replay_10y_hmmv4.yaml")

    assert config.start_date == "2016-01-01"
    assert config.end_date == "latest"
    assert config.train_lookback_days == 2520
    assert config.min_train_rows == 1260
    assert config.require_10y_replay is True
    assert config.allow_partial_backtest is False
    assert config.allow_silent_date_fallback is False
    assert "hmm_v4_path_aware_meta" in config.models


def test_strict_10y_replay_writes_preflight_failure_artifacts(tmp_path: Path) -> None:
    frame = _strict_replay_frame()
    feature_store_path = tmp_path / "features_daily.csv"
    frame.to_csv(feature_store_path, index=False)

    config_path = tmp_path / "hmm_replay_10y_hmmv4.yaml"
    config_path.write_text(
        "\n".join(
            [
                'start_date: "2016-01-01"',
                'end_date: "latest"',
                "train_lookback_days: 2520",
                "min_train_rows: 1260",
                "horizons: [1, 2, 3, 5, 10]",
                "models: [hmm_v4_path_aware_meta]",
                "output_dir: reports/backtests/hmm_replay_10y/",
                "artifact_dir: data/backtests/hmm_replay_10y/",
                f"feature_store_path: {feature_store_path.as_posix()}",
                "allow_partial_backtest: false",
                "allow_silent_date_fallback: false",
                "require_10y_replay: true",
            ]
        ),
        encoding="utf-8",
    )
    config = load_replay_config(config_path)
    config.output_dir = str(tmp_path / "reports")
    config.artifact_dir = str(tmp_path / "artifacts")

    try:
        run_hmm_replay(config=config)
        assert False, "Expected strict 10-year replay preflight to fail."
    except RuntimeError as exc:
        assert "Replay preflight failed." in str(exc)

    run_dirs = sorted((tmp_path / "reports").glob("run_*"))
    assert run_dirs
    report_path = run_dirs[0] / "preflight_failure_report.md"
    missing_path = run_dirs[0] / "preflight_missing_data.csv"
    assert report_path.exists()
    assert missing_path.exists()
    assert "2013-01-01" in report_path.read_text(encoding="utf-8")
