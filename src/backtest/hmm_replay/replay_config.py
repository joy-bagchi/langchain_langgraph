from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS = [
    "heuristic",
    "hmm_v1_core",
    "hmm_v2_core_plus_sector_corr",
    "hmm_v3_core_plus_sector_geometry",
    "hmm_v3_1_meta_blend",
]


@dataclass(slots=True)
class ReplayConfig:
    start_date: str
    end_date: str
    train_lookback_days: int = 756
    min_train_rows: int = 504
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3])
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    retrain_each_date: bool = True
    covariance_type: str = "diag"
    n_components: int = 4
    random_state: int = 42
    output_dir: str = "reports/backtests/hmm_replay/"
    artifact_dir: str = "data/backtests/hmm_replay/"
    feature_store_path: str = "agentic_vol_regime_app/data/processed/features_daily.parquet"
    freeze_policy_outputs: bool = True


def _to_int_list(values: list[Any]) -> list[int]:
    return [int(item) for item in list(values)]


def load_replay_config(path: str | Path) -> ReplayConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ReplayConfig(
        start_date=str(payload.get("start_date", "")),
        end_date=str(payload.get("end_date", "")),
        train_lookback_days=int(payload.get("train_lookback_days", 756)),
        min_train_rows=int(payload.get("min_train_rows", 504)),
        horizons=_to_int_list(payload.get("horizons", [1, 2, 3])),
        models=[str(item) for item in list(payload.get("models", list(DEFAULT_MODELS)))],
        retrain_each_date=bool(payload.get("retrain_each_date", True)),
        covariance_type=str(payload.get("covariance_type", "diag")),
        n_components=int(payload.get("n_components", 4)),
        random_state=int(payload.get("random_state", 42)),
        output_dir=str(payload.get("output_dir", "reports/backtests/hmm_replay/")),
        artifact_dir=str(payload.get("artifact_dir", "data/backtests/hmm_replay/")),
        feature_store_path=str(
            payload.get("feature_store_path", "agentic_vol_regime_app/data/processed/features_daily.parquet")
        ),
        freeze_policy_outputs=bool(payload.get("freeze_policy_outputs", True)),
    )
