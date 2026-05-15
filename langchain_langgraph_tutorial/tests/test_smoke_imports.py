from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = [ROOT / "src", ROOT / "advanced" / "src"]


@contextmanager
def _sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path = [entry for entry in sys.path if entry != str(path)]


def _load_module_from_path(source_root: Path, file_path: Path) -> None:
    module_name = ".".join(file_path.relative_to(source_root).with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    with _sys_path(source_root):
        spec.loader.exec_module(module)


def test_all_starter_modules_import() -> None:
    module_paths: list[tuple[Path, Path]] = []
    for source_root in SOURCE_ROOTS:
        module_paths.extend(
            (source_root, path)
            for path in source_root.rglob("*.py")
            if path.name != "__init__.py"
        )

    assert module_paths, "No starter modules found for smoke import test."

    for source_root, module_path in sorted(module_paths):
        _load_module_from_path(source_root, module_path)
