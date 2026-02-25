from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [ROOT / "src", ROOT / "advanced" / "src"]


def _load_module_from_path(file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_all_starter_modules_import() -> None:
    module_paths: list[Path] = []
    for source_root in SOURCE_ROOTS:
        module_paths.extend(
            path
            for path in source_root.rglob("*.py")
            if path.name != "__init__.py"
        )

    assert module_paths, "No starter modules found for smoke import test."

    for module_path in sorted(module_paths):
        _load_module_from_path(module_path)
