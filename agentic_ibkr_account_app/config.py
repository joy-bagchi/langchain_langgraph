"""Config and path helpers for the IBKR account app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @classmethod
    def default(cls) -> "AppPaths":
        return cls(root=Path(__file__).resolve().parent)
