"""Evaluation service contract and no-op default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_harness.shared.services import ServiceDescriptor


@dataclass(slots=True)
class EvaluationRequest:
    phase: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResponse:
    status: str = "skipped"
    score: float | None = None
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationService(Protocol):
    descriptor: ServiceDescriptor

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate runtime behavior or artifacts."""


class BasicEvaluationService:
    """Rule-based runtime evaluator for step and run outcomes."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="evaluation",
            implementation_id="basic_evaluation_service",
            maturity="simple",
            capabilities=["step_runtime_eval", "rule_based_findings"],
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        findings: list[str] = []
        status = "passed"
        score: float | None = 1.0
        request_status = str(request.payload.get("status", "")).strip().lower()
        if request_status in {"failed", "error", "rejected"}:
            status = "failed"
            score = 0.0
            findings.append(f"{request.phase} execution reported status '{request_status}'.")
        elif request_status in {"awaiting_review", "retrying"}:
            status = "attention_required"
            score = 0.5
            findings.append(f"{request.phase} execution requires follow-up: '{request_status}'.")
        return EvaluationResponse(
            status=status,
            score=score,
            findings=findings,
            metadata={"phase": request.phase},
        )

