"""Evaluation service contract and no-op default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_workflow.shared.services import ServiceDescriptor


@dataclass(slots=True)
class EvaluationRequest:
    phase: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResponse:
    status: str = "skipped"
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationService(Protocol):
    descriptor: ServiceDescriptor

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate runtime behavior or artifacts."""


class NullEvaluationService:
    """Placeholder evaluation service."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="evaluation",
            implementation_id="null_evaluation_service",
            maturity="simple",
            capabilities=[],
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        return EvaluationResponse()
