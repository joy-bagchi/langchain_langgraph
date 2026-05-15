"""Starter module: 01_state_and_nodes.

Implement lesson logic for this module.
"""

from __future__ import annotations

from typing import TypedDict
from langgraph.graph import StateGraph


class SharedState(TypedDict):
    """The state definition for the graph."""
    aggregate: list[str]


def node_1(state: SharedState) -> dict:
    """A simple node that adds to the state."""
    return {"aggregate": ["Hello"]}


def node_2(state: SharedState) -> dict:
    """A simple node that adds to the state."""
    return {"aggregate": ["World"]}


def main() -> None:
    """Run a minimal state and nodes graph."""
    # 1. Initialize the graph with the state schema
    builder = StateGraph(SharedState)

    # 2. Add nodes
    builder.add_node("model_node_1", node_1)
    builder.add_node("model_node_2", node_2)

    # 3. Define the flow
    builder.set_entry_point("model_node_1")
    builder.add_edge("model_node_1", "model_node_2")
    builder.set_finish_point("model_node_2")

    # 4. Compile and run
    graph = builder.compile()
    result = graph.invoke({"aggregate": []})
    print(f"Graph Result: {result}")


if __name__ == "__main__":
    main()
