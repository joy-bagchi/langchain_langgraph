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
    allowed_tools: list[str] = field(default_factory=list)
    memory_namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
        )


@dataclass(slots=True)
class MemoryQuery:
    """Memory lookup request."""

    namespace: str
    text: str
    max_results: int = 5
    memory_types: list[str] = field(default_factory=list)


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
    checkpoint_index: int
    last_error: str | None
