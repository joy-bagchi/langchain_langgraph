"""Standard cognitive service contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_workflow.contracts import WorkflowGraphState, WorkflowStep
from agentic_workflow.shared.services import ServiceDescriptor


@dataclass(slots=True)
class PromptExecutionRequest:
    """Prompt execution request sent to the cognitive layer."""

    prompt: str
    step: WorkflowStep
    state: WorkflowGraphState
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptExecutionResponse:
    """Prompt execution result returned by the cognitive layer."""

    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class CognitiveService(Protocol):
    """Microservice-like contract for prompt execution."""

    descriptor: ServiceDescriptor

    def execute_prompt(self, request: PromptExecutionRequest) -> PromptExecutionResponse:
        """Execute a prompt request."""


class DefaultCognitiveService:
    """Simple in-process cognitive service with optional model backing."""

    def __init__(self, *, model_callable=None) -> None:
        self.model_callable = model_callable
        self.descriptor = ServiceDescriptor(
            service_name="cognitive",
            implementation_id="default_cognitive_service",
            maturity="simple",
            capabilities=["prompt_execution", "deterministic_fallback"],
        )

    def execute_prompt(self, request: PromptExecutionRequest) -> PromptExecutionResponse:
        if self.model_callable is None:
            return PromptExecutionResponse(
                output=request.prompt,
                metadata={"mode": "deterministic"},
            )
        output = self.model_callable(request.prompt, request.step, request.state)
        return PromptExecutionResponse(
            output=output,
            metadata={"mode": "llm"},
        )
