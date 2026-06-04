from __future__ import annotations

import sys
from pathlib import Path


def _ensure_monorepo_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    harness_root = repo_root / "agentic_harness"
    for candidate in (str(repo_root), str(harness_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_monorepo_imports()
