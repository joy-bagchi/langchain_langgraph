"""Starter tests: Week 1 Day 5 prompt lab."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prompt_lab_module():
    root = Path(__file__).resolve().parents[1]
    file_path = root / "src" / "apps" / "prompt_lab.py"
    spec = importlib.util.spec_from_file_location("prompt_lab", file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompt_lab = _load_prompt_lab_module()


def test_build_parser_defaults() -> None:
    """Parser should expose stable defaults for the lab."""
    parser = prompt_lab.build_parser()
    args = parser.parse_args([])
    assert args.topic == "LangChain"
    assert args.mode == "explain"
    assert args.model == "openai:gpt-5.1"


def test_build_parser_mode_choices() -> None:
    """Parser should reject unsupported modes."""
    parser = prompt_lab.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "invalid-mode"])


def test_prompt_lab_scaffold_functions_exist() -> None:
    """Scaffold should provide key extension points for Day 5."""
    assert callable(prompt_lab.build_prompt)
    assert callable(prompt_lab.build_chain)
    assert callable(prompt_lab.run_once)
