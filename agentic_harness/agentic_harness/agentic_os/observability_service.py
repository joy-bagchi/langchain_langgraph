"""Observability service contract with local and optional LangSmith support."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agentic_harness.shared.services import ServiceDescriptor, ServiceEvent


@dataclass(slots=True)
class ObservabilityRequest:
    event: ServiceEvent


@dataclass(slots=True)
class LangSmithConfig:
    """Optional LangSmith tracing configuration."""

    enabled: bool = False
    api_key: str | None = None
    endpoint: str | None = None
    project: str | None = None
    workspace_id: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "project": self.project,
            "workspace_id": self.workspace_id,
        }


class ObservabilityService(Protocol):
    descriptor: ServiceDescriptor

    def record(self, request: ObservabilityRequest) -> dict:
        """Record an observability event and return its payload."""

    def trace_context(
        self,
        *,
        project_name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Return a context manager for optional distributed tracing."""

    def flush(self) -> None:
        """Flush buffered observability sinks."""


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_langsmith_config(
    *,
    enabled: bool | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
    project: str | None = None,
    workspace_id: str | None = None,
) -> LangSmithConfig:
    """Resolve LangSmith configuration from explicit args and environment."""
    resolved_enabled = _env_truthy("LANGSMITH_TRACING") if enabled is None else enabled
    return LangSmithConfig(
        enabled=bool(resolved_enabled),
        api_key=api_key or os.getenv("LANGSMITH_API_KEY"),
        endpoint=endpoint or os.getenv("LANGSMITH_ENDPOINT"),
        project=project or os.getenv("LANGSMITH_PROJECT") or "agentic-harness",
        workspace_id=workspace_id or os.getenv("LANGSMITH_WORKSPACE_ID"),
    )


class EventObservabilityService:
    """Local observability plus optional LangSmith tracing."""

    def __init__(
        self,
        *,
        langsmith_config: LangSmithConfig | None = None,
        langsmith_client: Any | None = None,
        tracing_context_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.langsmith_config = langsmith_config or resolve_langsmith_config()
        self._langsmith_client = None
        self._tracing_context_factory = tracing_context_factory
        metadata = {"sinks": ["local_events"]}

        if self.langsmith_config.enabled:
            metadata["sinks"].append("langsmith")
            metadata["langsmith"] = self.langsmith_config.to_metadata()
            if langsmith_client is not None:
                self._langsmith_client = langsmith_client
            elif self.langsmith_config.api_key:
                try:
                    from langsmith import Client
                    from langsmith.run_helpers import tracing_context

                    self._langsmith_client = Client(
                        api_key=self.langsmith_config.api_key,
                        api_url=self.langsmith_config.endpoint,
                        workspace_id=self.langsmith_config.workspace_id,
                    )
                    self._tracing_context_factory = tracing_context
                except Exception as exc:  # pragma: no cover - defensive fallback
                    metadata["langsmith_error"] = str(exc)
            else:
                metadata["langsmith_error"] = "LANGSMITH_API_KEY is not configured"

        self.descriptor = ServiceDescriptor(
            service_name="observability",
            implementation_id="event_observability_service",
            maturity="simple",
            capabilities=[
                "event_recording",
                *(
                    ["langsmith_tracing", "langsmith_flush"]
                    if self.langsmith_config.enabled and self._langsmith_client is not None
                    else []
                ),
            ],
            metadata=metadata,
        )

    def record(self, request: ObservabilityRequest) -> dict:
        payload = {"type": request.event.event_type, **request.event.payload}
        if self.langsmith_config.enabled and self._langsmith_client is not None and self.langsmith_config.project:
            payload["langsmith_project"] = self.langsmith_config.project
        return payload

    def trace_context(
        self,
        *,
        project_name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self.langsmith_config.enabled or self._langsmith_client is None or self._tracing_context_factory is None:
            return nullcontext()
        return self._tracing_context_factory(
            enabled=True,
            client=self._langsmith_client,
            project_name=project_name or self.langsmith_config.project,
            tags=tags or [],
            metadata=metadata or {},
        )

    def flush(self) -> None:
        if self._langsmith_client is not None:
            self._langsmith_client.flush()

