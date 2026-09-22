"""Immutable local baseline-history writer."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from typing import Any
from market_physics_core.adapters.baseline_store.artifacts import bundle_relative_path, serialize_bundle

def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def write_local_history(root: Path, summary: dict[str, Any], observations: list[dict[str, Any]]) -> tuple[Path, dict[str, str]]:
    run_dir = root / Path(bundle_relative_path(summary))
    if run_dir.exists():
        required = [run_dir / name for name in ("summary.json", "daily_skew.csv", "report.md")]
        if not all(path.is_file() for path in required):
            raise ValueError(f"incomplete existing immutable run: {run_dir}")
        return run_dir, {path.name: _sha(path) for path in required}
    run_dir.mkdir(parents=True, exist_ok=False)
    serialized, artifacts = serialize_bundle(summary, observations)
    summary.clear(); summary.update(serialized)
    for name, payload in artifacts.items():
        (run_dir / name).write_bytes(payload)
    return run_dir, {name: _sha(run_dir / name) for name in artifacts}

__all__ = ["write_local_history"]
