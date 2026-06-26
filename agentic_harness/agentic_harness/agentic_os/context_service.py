"""Context service contract and default implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_harness.context import ContextManager, ContextSnapshot
from agentic_harness.contracts import ContextPolicy, WorkflowDefinition, WorkflowGraphState, WorkflowStep
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

    def __init__(self, *, default_policy: ContextPolicy | None = None) -> None:
        self.default_policy = default_policy or ContextPolicy()
        self.descriptor = ServiceDescriptor(
            service_name="context",
            implementation_id="default_context_service",
            maturity="simple",
            capabilities=["assembly", "deterministic_compaction", "token_budgeting"],
        )

    def assemble_context(self, request: ContextServiceRequest) -> ContextSnapshot:
        policy = ContextPolicy(**dict(request.state.get("context_policy", {}))) if request.state.get("context_policy") else self.default_policy
        manager = ContextManager(
            workflow_definition=request.workflow_definition,
            policy=policy,
        )
        return manager.build_context(request.step, request.state)

