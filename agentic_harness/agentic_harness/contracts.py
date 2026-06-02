"""Typed contracts for the memory-aware workflow runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict
from uuid import uuid4


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def expires_at_from_ttl(ttl_days: int | None) -> str | None:
    """Convert a TTL expressed in days into an ISO-8601 timestamp."""
    if ttl_days is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()


def dataclass_dict(value: Any) -> Any:
    """Convert nested dataclasses into json-serializable dictionaries."""
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_dict(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class BranchRule:
    """Route to the next step when the expression evaluates truthy."""

    when: str
    next_step: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BranchRule":
        return cls(
            when=str(payload["when"]).strip(),
            next_step=str(payload["next"]).strip(),
        )


@dataclass(slots=True)
class MemoryWritePolicy:
    """Controls how a step emits durable memory records."""

    enabled: bool = False
    memory_type: str = "artifact_ref"
    namespace: str | None = None
    template: str | None = None
    ttl_days: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryWritePolicy":
        data = payload or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            memory_type=str(data.get("type", "artifact_ref")),
            namespace=data.get("namespace"),
            template=data.get("template"),
            ttl_days=data.get("ttl_days"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class WorkflowStep:
    """A single executable workflow step."""

    step_id: str
    title: str
    step_type: str
    output_key: str | None = None
    next_step: str | None = None
    branches: list[BranchRule] = field(default_factory=list)
    approved_next: str | None = None
    rejected_next: str | None = None
    prompt: str | None = None
    executor: str | None = None
    max_retries: int = 0
    memory: MemoryWritePolicy = field(default_factory=MemoryWritePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class WorkflowDefinition:
    """Top-level workflow configuration."""

    workflow_id: str
    title: str
    entry_step: str
    steps: dict[str, WorkflowStep]
    memory_namespace: str
    default_model: str | None = None
    description: str = ""
    workflow_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MockDefinition:
    """Mock execution contract for safe workflow testing."""

    enabled: bool = True
    response: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MockDefinition":
        data = payload or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            response=dict(data.get("response", {})),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class DeclarativeWorkflowNode:
    """Node in a declarative DAG-oriented workflow definition."""

    node_id: str
    kind: str
    purpose: str = ""
    agent: str | None = None
    depends_on: list[str] = field(default_factory=list)
    input_bindings: dict[str, Any] = field(default_factory=dict)
    artifact_contract: str | None = None
    execution_mode: str = "real"
    mock: MockDefinition | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeclarativeWorkflowDefinition:
    """Declarative business workflow targeted at future DAG compilation."""

    workflow_id: str
    title: str
    nodes: dict[str, DeclarativeWorkflowNode]
    entry_nodes: list[str]
    description: str = ""
    workflow_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowDagBlueprint:
    """Validated DAG blueprint derived from a declarative workflow."""

    workflow_id: str
    nodes: dict[str, DeclarativeWorkflowNode]
    roots: list[str]
    leaves: list[str]
    topological_order: list[str]
    adjacency: dict[str, list[str]]


@dataclass(slots=True)
class CompiledDagNode:
    """Executable node produced by the DAG compiler."""

    node_id: str
    kind: str
    purpose: str
    execution_mode: str
    agent: str | None = None
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    stage_index: int = 0
    artifact_contract: str | None = None
    input_bindings: dict[str, Any] = field(default_factory=dict)
    mock_response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompiledWorkflowDag:
    """Compiled DAG execution plan ready for runtime scheduling."""

    workflow_id: str
    title: str
    nodes: dict[str, CompiledDagNode]
    roots: list[str]
    leaves: list[str]
    topological_order: list[str]
    execution_stages: list[list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DagNodeExecutionRecord:
    """Internal execution ledger for a single DAG node run."""

    node_id: str
    kind: str
    status: str
    execution_mode: str
    inputs: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] | None = None
    child_run_id: str | None = None
    error: str | None = None
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class DagWorkflowRunResult:
    """Internal run state produced by the DAG executor."""

    run_id: str
    workflow_id: str
    workflow_title: str
    workflow_path: str | None
    status: str
    input_payload: dict[str, Any]
    node_results: dict[str, DagNodeExecutionRecord]
    artifacts: dict[str, dict[str, Any]]
    leaf_artifacts: dict[str, dict[str, Any]]
    execution_stages: list[list[str]]
    completed_stages: list[int] = field(default_factory=list)
    pending_human_gate: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class AgentDefinition:
    """Declarative agent contract bound to a workflow and service policy."""

    agent_id: str
    name: str
    role: str
    workflow_path: str
    description: str = ""
    llm_provider: str = "none"
    model: str | None = None
    temperature: float = 0.0
    memory_service_type: str = "filesystem"
    runtime_profile: str = "default"
    allowed_tools: list[str] = field(default_factory=list)
    memory_namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContextPolicy:
    """Rule-based policy that bounds assembled execution context."""

    max_recent_history: int = 3
    max_memory_hits: int = 3
    max_working_notes_chars: int = 500
    token_budget: int = 1200
    compaction_strategy: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class MemoryLifecyclePolicy:
    """Rule-based memory lifecycle policy owned by the OS."""

    namespace_strategy: str = "agent_or_workflow"
    max_ephemeral_records: int = 100
    max_durable_records: int = 5000
    consolidation_strategy: str = "merge_duplicates"
    suspend_on_idle: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class AgentRuntimeProfile:
    """Named runtime defaults applied by the OS to agent invocations."""

    profile_id: str = "default"
    context_policy: ContextPolicy = field(default_factory=ContextPolicy)
    memory_policy: MemoryLifecyclePolicy = field(default_factory=MemoryLifecyclePolicy)
    suspension_threshold_seconds: int = 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "context_policy": self.context_policy.to_dict(),
            "memory_policy": self.memory_policy.to_dict(),
            "suspension_threshold_seconds": self.suspension_threshold_seconds,
        }


@dataclass(slots=True)
class CompactionDecision:
    """OS decision record describing how context was compacted."""

    triggered: bool = False
    reason: str = ""
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    trimmed_history_items: int = 0
    trimmed_memory_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class AgentInvocation:
    """Durable invocation record for a running or resumable agent."""

    invocation_id: str
    run_id: str
    agent_id: str
    workflow_id: str
    status: str
    runtime_profile: str = "default"
    parent_run_id: str | None = None
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)


@dataclass(slots=True)
class ArtifactEnvelope:
    """Public machine-readable output produced by an agent or workflow."""

    artifact_type: str
    version: str
    producer: dict[str, Any]
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactEnvelope":
        return cls(
            artifact_type=str(payload["artifact_type"]),
            version=str(payload.get("version", "1.0")),
            producer=dict(payload.get("producer", {})),
            payload=dict(payload.get("payload", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ResponseEnvelope:
    """Audience-aware presentation layer for callers."""

    audience: str
    response_format: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResponseEnvelope":
        return cls(
            audience=str(payload.get("audience", "human")),
            response_format=str(payload.get("response_format", "json")),
            content=payload.get("content"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class StepExecutionResult:
    """Normalized result emitted by a step executor."""

    output: Any = None
    status: str = "succeeded"
    next_step: str | None = None
    awaiting_review: bool = False
    review_request: dict[str, Any] | None = None
    error: str | None = None
    memory_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepHistoryEntry:
    """Step-level execution journal entry."""

    step_id: str
    status: str
    output: Any
    next_step: str | None
    attempt: int
    timestamp: str = field(default_factory=utc_now)
    memory_record_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepHistoryEntry":
        return cls(
            step_id=payload["step_id"],
            status=payload["status"],
            output=payload.get("output"),
            next_step=payload.get("next_step"),
            attempt=int(payload.get("attempt", 0)),
            timestamp=payload.get("timestamp", utc_now()),
            memory_record_ids=list(payload.get("memory_record_ids", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class MemoryRecord:
    """A durable memory record stored on disk."""

    record_id: str
    namespace: str
    memory_type: str
    content: str
    source_run_id: str
    source_step_id: str
    created_at: str = field(default_factory=utc_now)
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    structured_payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        namespace: str,
        memory_type: str,
        content: str,
        source_run_id: str,
        source_step_id: str,
        ttl_days: int | None = None,
        metadata: dict[str, Any] | None = None,
        structured_payload: dict[str, Any] | None = None,
    ) -> "MemoryRecord":
        return cls(
            record_id=str(uuid4()),
            namespace=namespace,
            memory_type=memory_type,
            content=content,
            source_run_id=source_run_id,
            source_step_id=source_step_id,
            expires_at=expires_at_from_ttl(ttl_days),
            metadata=dict(metadata or {}),
            structured_payload=dict(structured_payload or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclass_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=payload["record_id"],
            namespace=payload["namespace"],
            memory_type=payload["memory_type"],
            content=payload["content"],
            source_run_id=payload["source_run_id"],
            source_step_id=payload["source_step_id"],
            created_at=payload.get("created_at", utc_now()),
            expires_at=payload.get("expires_at"),
            metadata=dict(payload.get("metadata", {})),
            structured_payload=dict(payload.get("structured_payload", {})),
        )


@dataclass(slots=True)
class MemoryQuery:
    """Memory lookup request."""

    namespace: str
    text: str
    max_results: int = 5
    memory_types: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    structured_filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemorySearchResult:
    """Returned match from the memory store."""

    record: MemoryRecord
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"record": self.record.to_dict(), "score": self.score}


class WorkflowGraphState(TypedDict, total=False):
    """State carried across the LangGraph workflow."""

    run_id: str
    agent_id: str | None
    agent_name: str | None
    agent_role: str | None
    invocation_id: str | None
    runtime_profile: str | None
    allowed_tools: list[str]
    workflow_id: str
    workflow_title: str
    workflow_path: str
    status: str
    current_step: str | None
    input_payload: dict[str, Any]
    named_outputs: dict[str, Any]
    step_outputs: dict[str, Any]
    working_notes: list[str]
    memory_hits: list[dict[str, Any]]
    step_history: list[dict[str, Any]]
    pending_review: dict[str, Any] | None
    review_responses: dict[str, dict[str, Any]]
    retry_counts: dict[str, int]
    events: list[dict[str, Any]]
    execution_outcome: dict[str, Any]
    active_context: dict[str, Any]
    context_policy: dict[str, Any]
    memory_policy: dict[str, Any]
    compaction_decision: dict[str, Any] | None
    checkpoint_index: int
    last_error: str | None
