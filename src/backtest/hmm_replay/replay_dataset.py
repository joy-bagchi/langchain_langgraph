from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_BASE_COLUMNS = {
    "date",
    "spy_close",
    "vix",
    "vvix",
    "realized_vol_5d",
    "realized_vol_21d",
}


def _resolve_feature_store_path(path: str | Path) -> tuple[Path, list[Path]]:
    raw = Path(path)
    repo_root = Path(__file__).resolve().parents[3]
    app_root = repo_root / "agentic_vol_regime_app"

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(repo_root / raw)
        candidates.append(app_root / raw)
    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    for candidate in deduped:
        if candidate.exists():
            return candidate, deduped
    return deduped[0], deduped


def load_feature_store(path: str | Path) -> pd.DataFrame:
    feature_path, checked_paths = _resolve_feature_store_path(path)
    if not feature_path.exists():
        checked = "; ".join(str(item) for item in checked_paths)
        raise FileNotFoundError(
            "Historical feature store not found. "
            f"configured='{path}'. checked=[{checked}]. "
            "Create/populate the feature store file or point Replay Config Path to a valid file."
        )
    if feature_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(feature_path)
    elif feature_path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(feature_path)
    else:
        raise RuntimeError(f"Unsupported feature-store format: {feature_path.suffix}")
    if "date" not in frame.columns:
        raise RuntimeError("Feature store must include a 'date' column.")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    missing = sorted(REQUIRED_BASE_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"Feature store missing required columns: {', '.join(missing)}")
    return frame.sort_values("date").reset_index(drop=True)


def filter_date_range(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RuntimeError(f"{context} missing feature columns: {', '.join(missing)}")


def train_slice(frame: pd.DataFrame, *, as_of_date: str, lookback_days: int) -> pd.DataFrame:
    as_of = pd.to_datetime(as_of_date).date()
    scoped = frame[frame["date"] <= as_of]
    if scoped.empty:
        return scoped
    return scoped.tail(int(lookback_days)).reset_index(drop=True)
