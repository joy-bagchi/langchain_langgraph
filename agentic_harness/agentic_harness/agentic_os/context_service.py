"""Context service contract and default implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_harness.context import ContextManager, ContextSnapshot
from agentic_harness.contracts import WorkflowDefinition, WorkflowGraphState, WorkflowStep
from agentic_harness.shared.services import ServiceDescriptor


@dataclass(slots=True)
class ContextServiceRequest:
    workflow_definition: WorkflowDefinition
    step: WorkflowStep
    state: WorkflowGraphState


class ContextService(Protocol):
    descriptor: ServiceDescriptor

    def assemble_context(self, request: ContextServiceRequest) -> ContextSnapshot:
        """Return the active context packet."""


class DefaultContextService:
    """Simple deterministic context-engineering service."""

    def __init__(self, *, max_recent_history: int = 3, max_memory_hits: int = 3) -> None:
        self.max_recent_history = max_recent_history
        self.max_memory_hits = max_memory_hits
        self.descriptor = ServiceDescriptor(
            service_name="context",
            implementation_id="default_context_service",
            maturity="simple",
            capabilities=["assembly", "deterministic_compaction"],
        )

    def assemble_context(self, request: ContextServiceRequest) -> ContextSnapshot:
        manager = ContextManager(
            workflow_definition=request.workflow_definition,
            max_recent_history=self.max_recent_history,
            max_memory_hits=self.max_memory_hits,
        )
        return manager.build_context(request.step, request.state)

