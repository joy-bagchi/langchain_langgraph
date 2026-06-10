"""Config and path helpers for the volatility regime app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path

    @property
    def agents_dir(self) -> Path:
        return self.root / "configs" / "agents"

    @property
    def workflows_dir(self) -> Path:
        return self.root / "configs" / "workflows"

    @property
    def thresholds_dir(self) -> Path:
        return self.root / "configs" / "thresholds"

    @property
    def features_dir(self) -> Path:
        return self.root / "configs" / "features"

    @property
    def sample_inputs_dir(self) -> Path:
        return self.root / "configs" / "sample_inputs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @classmethod
    def default(cls) -> "AppPaths":
        return cls(root=Path(__file__).resolve().parent)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
