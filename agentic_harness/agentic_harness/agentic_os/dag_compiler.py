"""Compile declarative workflows into executable DAG plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_harness.contracts import (
    CompiledDagNode,
    CompiledWorkflowDag,
    DeclarativeWorkflowDefinition,
    WorkflowDagBlueprint,
    dataclass_dict,
)
from agentic_harness.declarative_workflow import build_dag_blueprint
from agentic_harness.shared.services import ServiceDescriptor


WORKFLOW_REF_PATTERN = re.compile(r"^\$workflow\.[A-Za-z0-9_]+$")
NODE_ARTIFACT_REF_PATTERN = re.compile(r"^\$node\.([A-Za-z0-9_]+)\.artifact$")


def _resolve_execution_mode(node_kind: str, execution_mode: str, has_mock: bool) -> str:
    if node_kind == "mock_agent":
        return "mock"
    if execution_mode == "auto":
        return "mock" if has_mock and node_kind in {"agent", "service_task"} else "real"
    return execution_mode


def _collect_ancestor_nodes(
    node_id: str,
    nodes: dict[str, object],
    *,
    _cache: dict[str, set[str]] | None = None,
) -> set[str]:
    cache = _cache if _cache is not None else {}
    if node_id in cache:
        return cache[node_id]

    node = nodes[node_id]
    ancestors = set(getattr(node, "depends_on", []))
    for dependency in getattr(node, "depends_on", []):
        ancestors.update(_collect_ancestor_nodes(dependency, nodes, _cache=cache))
    cache[node_id] = ancestors
    return ancestors


def _validate_input_bindings(node_id: str, bindings: dict[str, object], accessible_nodes: set[str]) -> None:
    for binding_name, value in bindings.items():
        if not isinstance(value, str):
            continue
        if WORKFLOW_REF_PATTERN.match(value):
            continue
        node_match = NODE_ARTIFACT_REF_PATTERN.match(value)
        if node_match:
            dependency_id = node_match.group(1)
            if dependency_id not in accessible_nodes:
                raise ValueError(
                    f"Node '{node_id}' input binding '{binding_name}' references "
                    f"node '{dependency_id}' which is not available from its upstream DAG state."
                )
            continue
        raise ValueError(
            f"Node '{node_id}' input binding '{binding_name}' uses unsupported reference '{value}'."
        )


def _build_stage_indexes(blueprint: WorkflowDagBlueprint) -> dict[str, int]:
    stage_indexes: dict[str, int] = {}
    for node_id in blueprint.topological_order:
        node = blueprint.nodes[node_id]
        if not node.depends_on:
            stage_indexes[node_id] = 0
            continue
        stage_indexes[node_id] = max(stage_indexes[dependency] for dependency in node.depends_on) + 1
    return stage_indexes


def compile_workflow_dag(definition: DeclarativeWorkflowDefinition) -> CompiledWorkflowDag:
    """Compile a declarative workflow definition into an executable DAG plan."""
    blueprint = build_dag_blueprint(definition)
    stage_indexes = _build_stage_indexes(blueprint)

    compiled_nodes: dict[str, CompiledDagNode] = {}
    ancestor_cache: dict[str, set[str]] = {}
    for node_id in blueprint.topological_order:
        node = blueprint.nodes[node_id]
        accessible_nodes = _collect_ancestor_nodes(node_id, blueprint.nodes, _cache=ancestor_cache)
        _validate_input_bindings(node.node_id, node.input_bindings, accessible_nodes)
        compiled_nodes[node_id] = CompiledDagNode(
            node_id=node.node_id,
            kind=node.kind,
            purpose=node.purpose,
            execution_mode=_resolve_execution_mode(
                node.kind,
                node.execution_mode,
                has_mock=node.mock is not None,
            ),
            agent=node.agent,
            dependencies=list(node.depends_on),
            dependents=list(blueprint.adjacency[node_id]),
            stage_index=stage_indexes[node_id],
            artifact_contract=node.artifact_contract,
            input_bindings=dict(node.input_bindings),
            mock_response=dict(node.mock.response) if node.mock else {},
            metadata=dataclass_dict(node.metadata),
        )

    max_stage = max(stage_indexes.values(), default=0)
    execution_stages: list[list[str]] = [[] for _ in range(max_stage + 1)]
    for node_id, stage_index in stage_indexes.items():
        execution_stages[stage_index].append(node_id)
    execution_stages = [sorted(stage) for stage in execution_stages]

    return CompiledWorkflowDag(
        workflow_id=definition.workflow_id,
        title=definition.title,
        nodes=compiled_nodes,
        roots=list(blueprint.roots),
        leaves=list(blueprint.leaves),
        topological_order=list(blueprint.topological_order),
        execution_stages=execution_stages,
        metadata={
            "workflow_path": definition.workflow_path,
            "description": definition.description,
            **dict(definition.metadata),
        },
    )


@dataclass(slots=True)
class DefaultDagCompiler:
    """Simple in-process DAG compiler service for declarative workflows."""

    descriptor: ServiceDescriptor = field(
        default_factory=lambda: ServiceDescriptor(
            service_name="dag_compiler",
            implementation_id="default_dag_compiler",
            maturity="simple",
            capabilities=["dag_compilation", "stage_planning", "input_binding_validation"],
        )
    )

    def compile(self, definition: DeclarativeWorkflowDefinition) -> CompiledWorkflowDag:
        return compile_workflow_dag(definition)
