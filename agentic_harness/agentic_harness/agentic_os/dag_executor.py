"""Execute compiled DAG workflows through the agentic OS."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

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


class _StageExecutionState(TypedDict):
    """Transient LangGraph state for one DAG execution stage."""

    records: Annotated[list[dict[str, Any]], operator.add]


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


def _build_run_result(
    *,
    run_id: str,
    compiled: CompiledWorkflowDag,
    workflow_definition: DeclarativeWorkflowDefinition,
    status: str,
    workflow_input: dict[str, Any],
    node_results: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    execution_stages: list[list[str]],
    completed_stages: list[int],
    pending_human_gate: dict[str, Any] | None,
    events: list[dict[str, Any]],
    executor_id: str,
) -> dict[str, Any]:
    leaf_artifacts = {node_id: artifacts[node_id] for node_id in compiled.leaves if node_id in artifacts}
    return DagWorkflowRunResult(
        run_id=run_id,
        workflow_id=compiled.workflow_id,
        workflow_title=compiled.title,
        workflow_path=workflow_definition.workflow_path,
        status=status,
        input_payload=dict(workflow_input),
        node_results=node_results,
        artifacts=artifacts,
        leaf_artifacts=leaf_artifacts,
        execution_stages=execution_stages,
        completed_stages=completed_stages,
        pending_human_gate=pending_human_gate,
        events=events,
        metadata={
            "executor": executor_id,
            "topological_order": compiled.topological_order,
        },
    ).to_dict()


def _record_payload(record: DagNodeExecutionRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, DagNodeExecutionRecord):
        return record.to_dict()
    return dict(record)


def _execute_single_dag_node(
    node: CompiledDagNode,
    *,
    inputs: dict[str, Any],
    resolved_run_id: str,
    compiled: CompiledWorkflowDag,
    storage_root: str | Path | None,
    services,
    auto_approve_human_gates: bool,
) -> DagNodeExecutionRecord:
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
            record.status = "completed" if auto_approve_human_gates else "awaiting_human_gate"
            return record

        if node.execution_mode == "mock":
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
            return record

        if node.kind == "agent":
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
            record.status = (
                "completed"
                if child_result.get("status") == "completed"
                else child_result.get("status", "failed")
            )
            record.completed_at = utc_now()
            record.metadata["child_status"] = child_result.get("status")
            return record

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
        return record
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        record.completed_at = utc_now()
        return record


def _execute_stage_with_langgraph(
    active_nodes: list[tuple[str, CompiledDagNode, dict[str, Any]]],
    *,
    resolved_run_id: str,
    compiled: CompiledWorkflowDag,
    storage_root: str | Path | None,
    services,
    auto_approve_human_gates: bool,
) -> dict[str, DagNodeExecutionRecord]:
    """Execute one ready stage through a LangGraph fan-out/fan-in graph."""

    builder = StateGraph(_StageExecutionState)

    for node_id, node, inputs in active_nodes:
        def _worker(
            state: _StageExecutionState,
            *,
            _node: CompiledDagNode = node,
            _inputs: dict[str, Any] = inputs,
        ) -> dict[str, list[dict[str, Any]]]:
            record = _execute_single_dag_node(
                _node,
                inputs=_inputs,
                resolved_run_id=resolved_run_id,
                compiled=compiled,
                storage_root=storage_root,
                services=services,
                auto_approve_human_gates=auto_approve_human_gates,
            )
            return {"records": [record.to_dict()]}

        builder.add_node(node_id, _worker)
        builder.add_edge(START, node_id)
        builder.add_edge(node_id, END)

    graph = builder.compile()
    final_state = graph.invoke({"records": []})
    records = final_state.get("records", [])
    return {
        record["node_id"]: DagNodeExecutionRecord(**record)
        for record in records
    }


@dataclass(slots=True)
class DefaultDagExecutor:
    """Simple in-process executor for compiled DAG workflows."""

    descriptor: ServiceDescriptor = field(
        default_factory=lambda: ServiceDescriptor(
            service_name="dag_executor",
            implementation_id="default_dag_executor",
            maturity="simple",
            capabilities=[
                "langgraph_stage_parallelism",
                "agent_node_dispatch",
                "mock_nodes",
                "human_gate_pause",
            ],
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
        pending_ref = {"value": pending_human_gate}
        status = self._execute_from_stage(
            compiled,
            workflow_definition=workflow_definition,
            workflow_input=workflow_input,
            services=services,
            storage_root=storage_root,
            resolved_run_id=resolved_run_id,
            node_results=node_results,
            artifacts=artifacts,
            events=events,
            completed_stages=completed_stages,
            pending_human_gate_ref=pending_ref,
            start_stage_index=0,
            auto_approve_human_gates=auto_approve_human_gates,
        )
        pending_human_gate = pending_ref["value"]

        result = _build_run_result(
            run_id=resolved_run_id,
            compiled=compiled,
            workflow_definition=workflow_definition,
            status=status,
            workflow_input=workflow_input,
            node_results=node_results,
            artifacts=artifacts,
            execution_stages=compiled.execution_stages,
            completed_stages=completed_stages,
            pending_human_gate=pending_human_gate,
            events=events,
            executor_id=self.descriptor.implementation_id,
        )
        _persist_dag_state(storage_root, result)
        return result

    def _execute_from_stage(
        self,
        compiled: CompiledWorkflowDag,
        *,
        workflow_definition: DeclarativeWorkflowDefinition,
        workflow_input: dict[str, Any],
        services,
        storage_root: str | Path | None,
        resolved_run_id: str,
        node_results: dict[str, Any],
        artifacts: dict[str, dict[str, Any]],
        events: list[dict[str, Any]],
        completed_stages: list[int],
        pending_human_gate_ref: dict[str, Any],
        start_stage_index: int,
        auto_approve_human_gates: bool,
    ) -> str:
        status = "completed"
        for stage_index, stage in enumerate(compiled.execution_stages[start_stage_index:], start=start_stage_index):
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
            active_nodes: list[tuple[str, CompiledDagNode, dict[str, Any]]] = []
            for node_id in stage:
                node = compiled.nodes[node_id]
                existing_record = node_results.get(node_id)
                if isinstance(existing_record, dict) and existing_record.get("status") == "completed":
                    if node_id in artifacts:
                        continue
                active_nodes.append(
                    (
                        node_id,
                        node,
                        _build_node_inputs(
                            node,
                            workflow_input=workflow_input,
                            artifacts=artifacts,
                        ),
                    )
                )

            stage_results: dict[str, DagNodeExecutionRecord] = {}
            if active_nodes:
                stage_results = _execute_stage_with_langgraph(
                    active_nodes,
                    resolved_run_id=resolved_run_id,
                    compiled=compiled,
                    storage_root=storage_root,
                    services=services,
                    auto_approve_human_gates=auto_approve_human_gates,
                )

            stage_failed = False
            stage_paused = False
            for node_id in stage:
                record = stage_results.get(node_id)
                if record is None:
                    continue

                node_results[node_id] = _record_payload(record)
                if record.status == "completed" and record.artifact is not None:
                    artifacts[node_id] = dict(record.artifact)
                if record.status == "awaiting_human_gate":
                    stage_paused = True
                    if pending_human_gate_ref.get("value") is None:
                        pending_human_gate_ref["value"] = {
                            "node_id": record.node_id,
                            "purpose": compiled.nodes[node_id].purpose,
                            "artifact": record.artifact,
                        }
                    else:
                        current = dict(pending_human_gate_ref["value"])
                        current.setdefault("additional_gates", []).append(
                            {
                                "node_id": record.node_id,
                                "purpose": compiled.nodes[node_id].purpose,
                                "artifact": record.artifact,
                            }
                        )
                        pending_human_gate_ref["value"] = current
                    events.append(
                        {
                            "event_type": "human_gate_requested",
                            "payload": {
                                "timestamp": utc_now(),
                                "node_id": record.node_id,
                            },
                        }
                    )
                elif record.status != "completed":
                    stage_failed = True
                    events.append(
                        {
                            "event_type": "dag_node_failed",
                            "payload": {
                                "timestamp": utc_now(),
                                "node_id": record.node_id,
                                "error": record.error,
                                "status": record.status,
                            },
                        }
                    )
                else:
                    events.append(
                        {
                            "event_type": "dag_node_completed",
                            "payload": {
                                "timestamp": utc_now(),
                                "node_id": record.node_id,
                                "status": record.status,
                                "execution_mode": compiled.nodes[node_id].execution_mode,
                            },
                        }
                    )

            if stage_failed:
                return "failed"
            if stage_paused:
                return "awaiting_human_gate"

            if stage_index not in completed_stages:
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
        return status

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

    def resume(
        self,
        run_id: str,
        *,
        services,
        storage_root: str | Path | None = None,
        decision: str = "approved",
        notes: str | None = None,
        auto_approve_human_gates: bool = False,
    ) -> dict[str, Any]:
        storage_path = Path(storage_root or Path.cwd() / ".workflow_memory")
        run_store = WorkflowRunStore(storage_path)
        state = run_store.load_state(run_id)
        definition = services.declarative_workflow_definitions.load(state["workflow_path"])
        compiled = services.dag_compiler.compile(definition)

        pending_human_gate = dict(state.get("pending_human_gate") or {})
        if not pending_human_gate:
            raise ValueError(f"DAG run '{run_id}' is not waiting on a human gate.")

        decision_normalized = decision.strip().lower()
        if decision_normalized not in {"approved", "rejected"}:
            raise ValueError("Human gate decision must be 'approved' or 'rejected'.")

        node_id = str(pending_human_gate["node_id"])
        stage_index = compiled.nodes[node_id].stage_index
        node_results = dict(state.get("node_results", {}))
        artifacts = dict(state.get("artifacts", {}))
        events = list(state.get("events", []))
        completed_stages = list(state.get("completed_stages", []))

        gate_record = dict(node_results.get(node_id, {}))
        gate_record["completed_at"] = utc_now()
        gate_record["metadata"] = dict(gate_record.get("metadata", {}))
        gate_record["metadata"]["review_decision"] = decision_normalized
        if notes:
            gate_record["metadata"]["review_notes"] = notes

        if decision_normalized == "rejected":
            gate_record["status"] = "rejected"
            node_results[node_id] = gate_record
            events.append(
                {
                    "event_type": "human_gate_rejected",
                    "payload": {
                        "timestamp": utc_now(),
                        "node_id": node_id,
                    },
                }
            )
            result = _build_run_result(
                run_id=run_id,
                compiled=compiled,
                workflow_definition=definition,
                status="rejected",
                workflow_input=dict(state.get("input_payload", {})),
                node_results=node_results,
                artifacts=artifacts,
                execution_stages=compiled.execution_stages,
                completed_stages=completed_stages,
                pending_human_gate=None,
                events=events,
                executor_id=self.descriptor.implementation_id,
            )
            _persist_dag_state(storage_path, result)
            return result

        gate_record["status"] = "completed"
        node_results[node_id] = gate_record
        if gate_record.get("artifact"):
            artifacts[node_id] = dict(gate_record["artifact"])
        if stage_index not in completed_stages:
            completed_stages.append(stage_index)
        events.append(
            {
                "event_type": "human_gate_approved",
                "payload": {
                    "timestamp": utc_now(),
                    "node_id": node_id,
                },
            }
        )

        pending_ref = {"value": None}
        status = self._execute_from_stage(
            compiled,
            workflow_definition=definition,
            workflow_input=dict(state.get("input_payload", {})),
            services=services,
            storage_root=storage_path,
            resolved_run_id=run_id,
            node_results=node_results,
            artifacts=artifacts,
            events=events,
            completed_stages=completed_stages,
            pending_human_gate_ref=pending_ref,
            start_stage_index=stage_index + 1,
            auto_approve_human_gates=auto_approve_human_gates,
        )
        result = _build_run_result(
            run_id=run_id,
            compiled=compiled,
            workflow_definition=definition,
            status=status,
            workflow_input=dict(state.get("input_payload", {})),
            node_results=node_results,
            artifacts=artifacts,
            execution_stages=compiled.execution_stages,
            completed_stages=completed_stages,
            pending_human_gate=pending_ref["value"],
            events=events,
            executor_id=self.descriptor.implementation_id,
        )
        _persist_dag_state(storage_path, result)
        return result


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


def resume_declarative_workflow(
    run_id: str,
    *,
    storage_root: str | Path | None = None,
    services=None,
    decision: str = "approved",
    notes: str | None = None,
    auto_approve_human_gates: bool = False,
) -> dict[str, Any]:
    """Resume a declarative workflow run that is waiting on a human gate."""
    from agentic_harness.agentic_os.platform import build_platform_services

    service_bundle = services or build_platform_services(storage_root=storage_root)
    return service_bundle.dag_executor.resume(
        run_id,
        services=service_bundle,
        storage_root=storage_root,
        decision=decision,
        notes=notes,
        auto_approve_human_gates=auto_approve_human_gates,
    )
