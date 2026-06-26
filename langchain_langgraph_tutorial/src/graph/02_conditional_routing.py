"""Week 2 Day 2: conditional routing with reducer-style state updates."""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph


class SharedState(TypedDict):
    """State shared across graph nodes."""

    query: str
    route: str
    aggregate: Annotated[list[str], add]
    diagnostics: Annotated[list[str], add]


def intake_node(state: SharedState) -> dict:
    """Record the input query in reducer-backed logs."""
    return {
        "aggregate": [f"Received query: {state['query']}"],
        "diagnostics": ["intake_node"],
    }


def route_query(state: SharedState) -> Literal["math", "general"]:
    """Route to `math` when query appears calculation-focused."""
    query = state["query"].lower()
    math_tokens = ("+", "-", "*", "/", "calculate", "sum", "multiply")
    return "math" if any(token in query for token in math_tokens) else "general"


def math_node(state: SharedState) -> dict:
    """Handle math-like requests."""
    return {
        "route": "math",
        "aggregate": ["Math branch selected"],
        "diagnostics": ["math_node"],
    }


def general_node(state: SharedState) -> dict:
    """Handle non-math requests."""
    return {
        "route": "general",
        "aggregate": ["General branch selected"],
        "diagnostics": ["general_node"],
    }


def finalize_node(state: SharedState) -> dict:
    """Finalize the run and leave a deterministic terminal marker."""
    return {
        "aggregate": [f"Completed route: {state['route']}"],
        "diagnostics": ["finalize_node"],
    }


def build_graph():
    """Build a conditional graph with reducer-backed state fields."""
    builder = StateGraph(SharedState)

    builder.add_node("intake", intake_node)
    builder.add_node("math_node", math_node)
    builder.add_node("general_node", general_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("intake")
    builder.add_conditional_edges(
        "intake",
        route_query,
        {"math": "math_node", "general": "general_node"},
    )
    builder.add_edge("math_node", "finalize")
    builder.add_edge("general_node", "finalize")
    builder.set_finish_point("finalize")
    return builder.compile()


def run_query(query: str) -> SharedState:
    """Run graph once for a query and return final state."""
    graph = build_graph()
    return graph.invoke(
        {"query": query.strip(), "route": "", "aggregate": [], "diagnostics": []}
    )


def main() -> None:
    """Run two demo invocations to show branch behavior."""
    general_result = run_query("Explain what LangGraph is.")
    math_result = run_query("Please calculate 7 + 5.")

    print(f"General Result: {general_result}")
    print(f"Math Result: {math_result}")


if __name__ == "__main__":
    main()
