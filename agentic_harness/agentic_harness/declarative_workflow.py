"""YAML-backed declarative workflow definitions targeted at DAG compilation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import yaml

from agentic_harness.contracts import (
    DeclarativeWorkflowDefinition,
    DeclarativeWorkflowNode,
    MockDefinition,
    WorkflowDagBlueprint,
)


ALLOWED_NODE_KINDS = {"agent", "mock_agent", "human_gate", "service_task"}
ALLOWED_EXECUTION_MODES = {"real", "mock", "auto"}


def _resolve_agent_path(base_path: Path, agent_ref: str | None) -> str | None:
    if not agent_ref:
        return None
    candidate = Path(agent_ref)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_path.parent / candidate).resolve())


def _parse_node(node_payload: dict[str, Any], *, workflow_path: Path) -> DeclarativeWorkflowNode:
    node_id = str(node_payload["id"]).strip()
    kind = str(node_payload["kind"]).strip()
    execution_mode = str(node_payload.get("execution_mode", "real")).strip().lower()

    return DeclarativeWorkflowNode(
        node_id=node_id,
        kind=kind,
        purpose=str(node_payload.get("purpose", "")).strip(),
        agent=_resolve_agent_path(workflow_path, node_payload.get("agent")),
        depends_on=[str(item).strip() for item in node_payload.get("depends_on", [])],
        input_bindings=dict(node_payload.get("input_bindings", {})),
        artifact_contract=node_payload.get("artifact_contract"),
        execution_mode=execution_mode,
        mock=MockDefinition.from_dict(node_payload.get("mock")) if node_payload.get("mock") else None,
        metadata={
            key: value
            for key, value in node_payload.items()
            if key
            not in {
                "id",
                "kind",
                "purpose",
                "agent",
                "depends_on",
                "input_bindings",
                "artifact_contract",
                "execution_mode",
                "mock",
            }
        },
    )


def _validate_definition(definition: DeclarativeWorkflowDefinition) -> None:
    if not definition.nodes:
        raise ValueError("Declarative workflow must define at least one node.")

    for entry_node in definition.entry_nodes:
        if entry_node not in definition.nodes:
            raise ValueError(f"Entry node '{entry_node}' is not defined in the workflow.")

    for node in definition.nodes.values():
        if node.kind not in ALLOWED_NODE_KINDS:
            raise ValueError(
                f"Node '{node.node_id}' has unsupported kind '{node.kind}'. "
                f"Supported kinds: {', '.join(sorted(ALLOWED_NODE_KINDS))}."
            )
        if node.execution_mode not in ALLOWED_EXECUTION_MODES:
            raise ValueError(
                f"Node '{node.node_id}' has unsupported execution_mode '{node.execution_mode}'. "
                f"Supported modes: {', '.join(sorted(ALLOWED_EXECUTION_MODES))}."
            )
        if node.kind == "agent" and not node.agent:
            raise ValueError(f"Agent node '{node.node_id}' must declare an 'agent' reference.")
        if node.kind == "mock_agent" and node.execution_mode == "real":
            raise ValueError(
                f"Mock agent node '{node.node_id}' cannot use execution_mode='real'."
            )
        if node.execution_mode == "mock" and node.mock is None:
            raise ValueError(
                f"Node '{node.node_id}' uses execution_mode='mock' but does not define a mock response."
            )
        if node.kind == "mock_agent" and node.mock is None:
            raise ValueError(
                f"Mock agent node '{node.node_id}' must define a mock response."
            )
        for dependency in node.depends_on:
            if dependency not in definition.nodes:
                raise ValueError(
                    f"Node '{node.node_id}' depends on unknown node '{dependency}'."
                )

    blueprint = build_dag_blueprint(definition)
    if set(blueprint.roots) != set(definition.entry_nodes):
        raise ValueError(
            "Entry nodes must exactly match the dependency-free root nodes of the workflow."
        )


def parse_declarative_workflow(
    payload: dict[str, Any],
    *,
    workflow_path: str | None = None,
) -> DeclarativeWorkflowDefinition:
    """Parse a YAML payload into a declarative workflow definition."""
    resolved_path = Path(workflow_path).resolve() if workflow_path else Path.cwd()
    workflow_id = str(payload["workflow_id"]).strip()
    title = str(payload.get("title", workflow_id)).strip()
    description = str(payload.get("description", "")).strip()

    nodes_list = payload.get("nodes", [])
    if not isinstance(nodes_list, list):
        raise ValueError("'nodes' must be a list in a declarative workflow definition.")

    nodes: dict[str, DeclarativeWorkflowNode] = {}
    for node_payload in nodes_list:
        node = _parse_node(dict(node_payload), workflow_path=resolved_path)
        if node.node_id in nodes:
            raise ValueError(f"Duplicate declarative workflow node id '{node.node_id}'.")
        nodes[node.node_id] = node

    inferred_entry_nodes = sorted(
        node.node_id for node in nodes.values() if not node.depends_on
    )
    entry_nodes = [str(item).strip() for item in payload.get("entry_nodes", inferred_entry_nodes)]

    definition = DeclarativeWorkflowDefinition(
        workflow_id=workflow_id,
        title=title,
        nodes=nodes,
        entry_nodes=entry_nodes,
        description=description,
        workflow_path=str(resolved_path) if workflow_path else None,
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"workflow_id", "title", "description", "entry_nodes", "nodes"}
        },
    )
    _validate_definition(definition)
    return definition


def load_declarative_workflow_definition(path: str | Path) -> DeclarativeWorkflowDefinition:
    """Load a declarative workflow definition from a YAML file."""
    workflow_path = Path(path).resolve()
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
    return parse_declarative_workflow(payload, workflow_path=str(workflow_path))


def build_dag_blueprint(definition: DeclarativeWorkflowDefinition) -> WorkflowDagBlueprint:
    """Build a validated DAG blueprint for future workflow compilation."""
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in definition.nodes}
    in_degree: dict[str, int] = {node_id: 0 for node_id in definition.nodes}

    for node in definition.nodes.values():
        for dependency in node.depends_on:
            adjacency[dependency].append(node.node_id)
            in_degree[node.node_id] += 1

    roots = sorted(node_id for node_id, degree in in_degree.items() if degree == 0)
    leaves = sorted(node_id for node_id, children in adjacency.items() if not children)

    queue: deque[str] = deque(sorted(roots))
    local_in_degree = dict(in_degree)
    topological_order: list[str] = []

    while queue:
        node_id = queue.popleft()
        topological_order.append(node_id)
        for child in sorted(adjacency[node_id]):
            local_in_degree[child] -= 1
            if local_in_degree[child] == 0:
                queue.append(child)

    if len(topological_order) != len(definition.nodes):
        raise ValueError(
            f"Declarative workflow '{definition.workflow_id}' contains a cycle and cannot be compiled into a DAG."
        )

    return WorkflowDagBlueprint(
        workflow_id=definition.workflow_id,
        nodes=dict(definition.nodes),
        roots=roots,
        leaves=leaves,
        topological_order=topological_order,
        adjacency={key: sorted(value) for key, value in adjacency.items()},
    )
