"""Execute compiled DAG workflows through the agentic OS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentic_harness.agentic_os.dag_compiler import compile_workflow_dag
from agentic_harness.contracts import (
    ArtifactEnvelope,
    CompiledDagNode,
    CompiledWorkflowDag,
    DagNodeExecutionRecord,
    DagWorkflowRunResult,
    DeclarativeWorkflowDefinition,
    utc_now,
)
from agentic_harness.outputs import extract_artifact
from agentic_harness.shared.services import ServiceDescriptor
from agentic_harness.stores import WorkflowRunStore


def _resolve_binding(
    value: Any,
    *,
    workflow_input: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> Any:
    if isinstance(value, str):
        if value.startswith("$workflow."):
            return workflow_input.get(value.removeprefix("$workflow."))
        if value.startswith("$node.") and value.endswith(".artifact"):
            node_id = value[len("$node.") : -len(".artifact")]
            return artifacts.get(node_id)
        return value
    if isinstance(value, list):
        return [_resolve_binding(item, workflow_input=workflow_input, artifacts=artifacts) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_binding(item, workflow_input=workflow_input, artifacts=artifacts)
            for key, item in value.items()
        }
    return value


def _build_node_inputs(
    node: CompiledDagNode,
    *,
    workflow_input: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        key: _resolve_binding(value, workflow_input=workflow_input, artifacts=artifacts)
        for key, value in node.input_bindings.items()
    }


def _persist_dag_state(storage_root: str | Path | None, state: dict[str, Any]) -> None:
    storage_path = Path(storage_root or Path.cwd() / ".workflow_memory")
    run_store = WorkflowRunStore(storage_path)
    run_store.save_state(state)
    run_store.save_manifest(
        state["run_id"],
        {
            "workflow_id": state["workflow_id"],
            "workflow_path": state.get("workflow_path"),
            "storage_root": str(storage_path),
            "execution_type": "declarative_workflow_dag",
        },
    )


@dataclass(slots=True)
class DefaultDagExecutor:
    """Simple in-process executor for compiled DAG workflows."""

    descriptor: ServiceDescriptor = field(
        default_factory=lambda: ServiceDescriptor(
            service_name="dag_executor",
            implementation_id="default_dag_executor",
            maturity="simple",
            capabilities=["sequential_stage_execution", "agent_node_dispatch", "mock_nodes", "human_gate_pause"],
        )
    )

    def execute(
        self,
        compiled: CompiledWorkflowDag,
        *,
        workflow_definition: DeclarativeWorkflowDefinition,
        workflow_input: dict[str, Any],
        services,
        storage_root: str | Path | None = None,
        run_id: str | None = None,
        auto_approve_human_gates: bool = False,
    ) -> dict[str, Any]:
        resolved_run_id = run_id or str(uuid4())
        node_results: dict[str, DagNodeExecutionRecord] = {}
        artifacts: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = [
            {
                "event_type": "dag_run_started",
                "payload": {
                    "timestamp": utc_now(),
                    "workflow_id": compiled.workflow_id,
                    "run_id": resolved_run_id,
                },
            }
        ]
        completed_stages: list[int] = []
        pending_human_gate: dict[str, Any] | None = None
        status = "completed"

        for stage_index, stage in enumerate(compiled.execution_stages):
            events.append(
                {
                    "event_type": "dag_stage_started",
                    "payload": {
                        "timestamp": utc_now(),
                        "stage_index": stage_index,
                        "node_ids": list(stage),
                    },
                }
            )
            for node_id in stage:
                node = compiled.nodes[node_id]
                inputs = _build_node_inputs(
                    node,
                    workflow_input=workflow_input,
                    artifacts=artifacts,
                )
                record = DagNodeExecutionRecord(
                    node_id=node.node_id,
                    kind=node.kind,
                    status="running",
                    execution_mode=node.execution_mode,
                    inputs=inputs,
                    metadata={
                        "purpose": node.purpose,
                        "artifact_contract": node.artifact_contract,
                        "stage_index": node.stage_index,
                    },
                )

                try:
                    if node.kind == "human_gate":
                        review_packet = ArtifactEnvelope(
                            artifact_type="review_packet",
                            version="1.0",
                            producer={
                                "workflow_id": compiled.workflow_id,
                                "run_id": resolved_run_id,
                                "node_id": node.node_id,
                            },
                            payload=inputs,
                            metadata={"status": "awaiting_human_gate"},
                        ).to_dict()
                        record.artifact = review_packet
                        record.completed_at = utc_now()
                        if auto_approve_human_gates:
                            record.status = "completed"
                            artifacts[node_id] = review_packet
                        else:
                            record.status = "awaiting_human_gate"
                            pending_human_gate = {
                                "node_id": node.node_id,
                                "purpose": node.purpose,
                                "artifact": review_packet,
                            }
                            node_results[node_id] = record
                            status = "awaiting_human_gate"
                            events.append(
                                {
                                    "event_type": "human_gate_requested",
                                    "payload": {
                                        "timestamp": utc_now(),
                                        "node_id": node.node_id,
                                    },
                                }
                            )
                            result = DagWorkflowRunResult(
                                run_id=resolved_run_id,
                                workflow_id=compiled.workflow_id,
                                workflow_title=compiled.title,
                                workflow_path=workflow_definition.workflow_path,
                                status=status,
                                input_payload=dict(workflow_input),
                                node_results=node_results,
                                artifacts=artifacts,
                                leaf_artifacts={},
                                execution_stages=compiled.execution_stages,
                                completed_stages=completed_stages,
                                pending_human_gate=pending_human_gate,
                                events=events,
                                metadata={
                                    "executor": self.descriptor.implementation_id,
                                    "topological_order": compiled.topological_order,
                                },
                            ).to_dict()
                            _persist_dag_state(storage_root, result)
                            return result
                    elif node.execution_mode == "mock":
                        artifact = dict(node.mock_response)
                        if "artifact_type" not in artifact:
                            artifact = ArtifactEnvelope(
                                artifact_type=node.artifact_contract or f"{node.node_id}_artifact",
                                version="1.0",
                                producer={
                                    "workflow_id": compiled.workflow_id,
                                    "run_id": resolved_run_id,
                                    "node_id": node.node_id,
                                },
                                payload=artifact,
                            ).to_dict()
                        record.artifact = artifact
                        record.status = "completed"
                        record.completed_at = utc_now()
                        artifacts[node_id] = artifact
                    elif node.kind == "agent":
                        from agentic_harness.runtime import run_agent_workflow

                        child_run_id = f"{resolved_run_id}__{node.node_id}"
                        child_result = run_agent_workflow(
                            node.agent or "",
                            inputs,
                            run_id=child_run_id,
                            storage_root=storage_root,
                            services=services,
                        )
                        artifact = extract_artifact(child_result).to_dict()
                        record.child_run_id = child_result.get("run_id")
                        record.artifact = artifact
                        record.status = "completed" if child_result.get("status") == "completed" else child_result.get("status", "failed")
                        record.completed_at = utc_now()
                        record.metadata["child_status"] = child_result.get("status")
                        artifacts[node_id] = artifact
                        if record.status != "completed":
                            status = record.status
                            node_results[node_id] = record
                            events.append(
                                {
                                    "event_type": "dag_node_failed",
                                    "payload": {
                                        "timestamp": utc_now(),
                                        "node_id": node.node_id,
                                        "status": record.status,
                                    },
                                }
                            )
                            result = DagWorkflowRunResult(
                                run_id=resolved_run_id,
                                workflow_id=compiled.workflow_id,
                                workflow_title=compiled.title,
                                workflow_path=workflow_definition.workflow_path,
                                status=status,
                                input_payload=dict(workflow_input),
                                node_results=node_results,
                                artifacts=artifacts,
                                leaf_artifacts={},
                                execution_stages=compiled.execution_stages,
                                completed_stages=completed_stages,
                                events=events,
                                metadata={
                                    "executor": self.descriptor.implementation_id,
                                    "topological_order": compiled.topological_order,
                                },
                            ).to_dict()
                            _persist_dag_state(storage_root, result)
                            return result
                    else:
                        artifact = ArtifactEnvelope(
                            artifact_type=node.artifact_contract or f"{node.node_id}_artifact",
                            version="1.0",
                            producer={
                                "workflow_id": compiled.workflow_id,
                                "run_id": resolved_run_id,
                                "node_id": node.node_id,
                            },
                            payload=inputs,
                        ).to_dict()
                        record.artifact = artifact
                        record.status = "completed"
                        record.completed_at = utc_now()
                        artifacts[node_id] = artifact
                except Exception as exc:
                    record.status = "failed"
                    record.error = str(exc)
                    record.completed_at = utc_now()
                    node_results[node_id] = record
                    status = "failed"
                    events.append(
                        {
                            "event_type": "dag_node_failed",
                            "payload": {
                                "timestamp": utc_now(),
                                "node_id": node.node_id,
                                "error": str(exc),
                            },
                        }
                    )
                    result = DagWorkflowRunResult(
                        run_id=resolved_run_id,
                        workflow_id=compiled.workflow_id,
                        workflow_title=compiled.title,
                        workflow_path=workflow_definition.workflow_path,
                        status=status,
                        input_payload=dict(workflow_input),
                        node_results=node_results,
                        artifacts=artifacts,
                        leaf_artifacts={},
                        execution_stages=compiled.execution_stages,
                        completed_stages=completed_stages,
                        events=events,
                        metadata={
                            "executor": self.descriptor.implementation_id,
                            "topological_order": compiled.topological_order,
                        },
                    ).to_dict()
                    _persist_dag_state(storage_root, result)
                    return result

                node_results[node_id] = record
                events.append(
                    {
                        "event_type": "dag_node_completed",
                        "payload": {
                            "timestamp": utc_now(),
                            "node_id": node.node_id,
                            "status": record.status,
                            "execution_mode": node.execution_mode,
                        },
                    }
                )

            completed_stages.append(stage_index)
            events.append(
                {
                    "event_type": "dag_stage_completed",
                    "payload": {
                        "timestamp": utc_now(),
                        "stage_index": stage_index,
                    },
                }
            )

        leaf_artifacts = {node_id: artifacts[node_id] for node_id in compiled.leaves if node_id in artifacts}
        result = DagWorkflowRunResult(
            run_id=resolved_run_id,
            workflow_id=compiled.workflow_id,
            workflow_title=compiled.title,
            workflow_path=workflow_definition.workflow_path,
            status=status,
            input_payload=dict(workflow_input),
            node_results=node_results,
            artifacts=artifacts,
            leaf_artifacts=leaf_artifacts,
            execution_stages=compiled.execution_stages,
            completed_stages=completed_stages,
            pending_human_gate=pending_human_gate,
            events=events,
            metadata={
                "executor": self.descriptor.implementation_id,
                "topological_order": compiled.topological_order,
            },
        ).to_dict()
        _persist_dag_state(storage_root, result)
        return result

    def execute_definition(
        self,
        definition: DeclarativeWorkflowDefinition,
        *,
        workflow_input: dict[str, Any],
        services,
        storage_root: str | Path | None = None,
        run_id: str | None = None,
        auto_approve_human_gates: bool = False,
    ) -> dict[str, Any]:
        compiled = compile_workflow_dag(definition)
        return self.execute(
            compiled,
            workflow_definition=definition,
            workflow_input=workflow_input,
            services=services,
            storage_root=storage_root,
            run_id=run_id,
            auto_approve_human_gates=auto_approve_human_gates,
        )


def run_declarative_workflow(
    workflow_path: str | Path,
    workflow_input: dict[str, Any],
    *,
    storage_root: str | Path | None = None,
    run_id: str | None = None,
    services=None,
    auto_approve_human_gates: bool = False,
) -> dict[str, Any]:
    """Load, compile, and execute a declarative workflow through the DAG executor."""
    from agentic_harness.agentic_os.platform import build_platform_services

    service_bundle = services or build_platform_services(storage_root=storage_root)
    definition = service_bundle.declarative_workflow_definitions.load(workflow_path)
    return service_bundle.dag_executor.execute_definition(
        definition,
        workflow_input=workflow_input,
        services=service_bundle,
        storage_root=storage_root,
        run_id=run_id,
        auto_approve_human_gates=auto_approve_human_gates,
    )
