"""Monorepo import bootstrap for local development."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_imports() -> None:
    """Expose sibling project packages when running directly from the monorepo."""
    repo_root = Path(__file__).resolve().parent.parent
    harness_project_root = repo_root / "agentic_harness"
    if harness_project_root.exists():
        candidate = str(harness_project_root)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
