from __future__ import annotations

import sys
from pathlib import Path


def _ensure_script_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "agentic_vol_regime_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_script_imports()

from agentic_vol_regime_app.overwrite_candidate_scorer import main


if __name__ == "__main__":
    raise SystemExit(main())
