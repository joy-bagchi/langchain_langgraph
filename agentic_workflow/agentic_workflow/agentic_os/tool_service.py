"""Tool service contract and inert default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_workflow.shared.services import ServiceDescriptor


@dataclass(slots=True)
class ToolExecutionRequest:
    tool_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionResponse:
    status: str
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolService(Protocol):
    descriptor: ServiceDescriptor

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a registered tool."""


class NullToolService:
    """Placeholder tool implementation for the first layered slice."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="tools",
            implementation_id="null_tool_service",
            maturity="simple",
            capabilities=[],
        )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        return ToolExecutionResponse(
            status="unavailable",
            metadata={"reason": "tool service not configured"},
        )
