"""Observability service contract and default event implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentic_harness.shared.services import ServiceDescriptor, ServiceEvent


@dataclass(slots=True)
class ObservabilityRequest:
    event: ServiceEvent


class ObservabilityService(Protocol):
    descriptor: ServiceDescriptor

    def record(self, request: ObservabilityRequest) -> dict:
        """Record an observability event and return its payload."""


class EventObservabilityService:
    """Simple observability implementation that normalizes events."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="observability",
            implementation_id="event_observability_service",
            maturity="simple",
            capabilities=["event_recording"],
        )

    def record(self, request: ObservabilityRequest) -> dict:
        return {"type": request.event.event_type, **request.event.payload}

