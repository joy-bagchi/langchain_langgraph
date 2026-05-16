"""Guardrail service contract and pass-through default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_harness.contracts import WorkflowGraphState, WorkflowStep
from agentic_harness.shared.services import GuardrailDecision, ServiceDescriptor


@dataclass(slots=True)
class GuardrailRequest:
    phase: str
    step: WorkflowStep
    state: WorkflowGraphState
    candidate_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GuardrailService(Protocol):
    descriptor: ServiceDescriptor

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate pre-step or post-step guardrails."""


class PassthroughGuardrailService:
    """Simple allow-all guardrail implementation."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="guardrails",
            implementation_id="passthrough_guardrail_service",
            maturity="simple",
            capabilities=["pre_step", "post_step"],
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return GuardrailDecision(allowed=True, action="allow")

