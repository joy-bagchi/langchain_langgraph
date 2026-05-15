"""Week 2 tests for state, nodes, and conditional routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_graph_module(file_name: str, module_name: str):
    root = Path(__file__).resolve().parents[1]
    file_path = root / "src" / "graph" / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


state_and_nodes = _load_graph_module("01_state_and_nodes.py", "state_and_nodes")
conditional_routing = _load_graph_module(
    "02_conditional_routing.py", "conditional_routing"
)


def test_linear_graph_nodes_append_expected_values() -> None:
    """Week 2 Day 1 nodes should contribute deterministic values."""
    assert state_and_nodes.node_1({"aggregate": []}) == {"aggregate": ["Hello"]}
    assert state_and_nodes.node_2({"aggregate": []}) == {"aggregate": ["World"]}


def test_conditional_routing_general_branch() -> None:
    """Non-math prompt should route through the general branch."""
    result = conditional_routing.run_query("Explain what prompt templates are")
    assert result["route"] == "general"
    assert "General branch selected" in result["aggregate"]
    assert result["diagnostics"] == ["intake_node", "general_node", "finalize_node"]


def test_conditional_routing_math_branch() -> None:
    """Math-like prompt should route through the math branch."""
    result = conditional_routing.run_query("Please calculate 9 + 2")
    assert result["route"] == "math"
    assert "Math branch selected" in result["aggregate"]
    assert result["diagnostics"] == ["intake_node", "math_node", "finalize_node"]
