"""Standard service contracts used across the layered platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ServiceDescriptor:
    """Describes a concrete service implementation."""

    service_name: str
    implementation_id: str
    maturity: str = "simple"
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuardrailDecision:
    """Result of a guardrail evaluation."""

    allowed: bool = True
    action: str = "allow"
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServiceEvent:
    """Normalized service-originated event."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
