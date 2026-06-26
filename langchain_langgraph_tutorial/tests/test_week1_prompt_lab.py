"""Starter tests: Week 1 Day 5 prompt lab."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

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


def test_build_prompt_supports_modes() -> None:
    """Day 5 prompt lab should provide both explain and quiz prompts."""
    explain_prompt = prompt_lab.build_prompt("explain")
    quiz_prompt = prompt_lab.build_prompt("quiz")

    assert "{topic}" in explain_prompt.messages[1].prompt.template
    assert "{topic}" in quiz_prompt.messages[1].prompt.template


def test_build_prompt_rejects_invalid_mode() -> None:
    """Prompt creation should fail fast for unsupported modes."""
    with pytest.raises(ValueError):
        prompt_lab.build_prompt("invalid")


def test_run_once_invokes_chain_with_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_once` should validate topic and invoke chain payload correctly."""

    class FakeChain:
        def __init__(self) -> None:
            self.received: dict[str, Any] | None = None

        def invoke(self, payload: dict[str, Any]) -> str:
            self.received = payload
            return "ok"

    fake_chain = FakeChain()

    def fake_build_chain(*, mode: str, model_name: str) -> FakeChain:
        assert mode == "explain"
        assert model_name == "openai:gpt-5.1"
        return fake_chain

    monkeypatch.setattr(prompt_lab, "build_chain", fake_build_chain)
    result = prompt_lab.run_once("  LangGraph  ", "explain", "openai:gpt-5.1")

    assert result == "ok"
    assert fake_chain.received == {"topic": "LangGraph"}


def test_run_once_rejects_empty_topic() -> None:
    """Topic validation should prevent blank inputs."""
    with pytest.raises(ValueError):
        prompt_lab.run_once("   ", "explain", "openai:gpt-5.1")
