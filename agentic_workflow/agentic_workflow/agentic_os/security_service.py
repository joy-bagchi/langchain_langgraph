"""Security/identity service contract and permissive default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_workflow.shared.services import ServiceDescriptor


@dataclass(slots=True)
class AuthorizationRequest:
    capability: str
    subject: str = "local_runner"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AuthorizationResponse:
    allowed: bool = True
    reason: str = "local_default_allow"
    metadata: dict[str, Any] = field(default_factory=dict)


class SecurityService(Protocol):
    descriptor: ServiceDescriptor

    def authorize(self, request: AuthorizationRequest) -> AuthorizationResponse:
        """Authorize a runtime capability."""


class PermissiveSecurityService:
    """Simple local authorization service."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="security",
            implementation_id="permissive_security_service",
            maturity="simple",
            capabilities=["local_authorization"],
        )

    def authorize(self, request: AuthorizationRequest) -> AuthorizationResponse:
        return AuthorizationResponse()
