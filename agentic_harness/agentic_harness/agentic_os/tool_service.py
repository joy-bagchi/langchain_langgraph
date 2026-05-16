"""Tool service contract and default registered tool implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agentic_harness.shared.services import ServiceDescriptor


@dataclass(slots=True)
class ToolDefinition:
    """Declarative description of a registered tool."""

    tool_id: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def list_tools(self) -> list[ToolDefinition]:
        """List registered tool definitions."""


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

    def list_tools(self) -> list[ToolDefinition]:
        return []


class TavilyWebSearchClient:
    """Thin adapter around Tavily for web search."""

    def __init__(self, *, api_key: str | None = None) -> None:
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ImportError(
                "tavily-python is required for the web_search tool."
            ) from exc

        resolved_api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not resolved_api_key:
            raise ValueError("TAVILY_API_KEY is required for the web_search tool.")
        self.client = TavilyClient(api_key=resolved_api_key)

    def search(
        self,
        *,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        return self.client.search(
            query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
        )


ToolHandler = Callable[[ToolExecutionRequest], ToolExecutionResponse]


class RegisteredToolService:
    """Toolbox implementation backed by a local registry of handlers."""

    def __init__(
        self,
        *,
        definitions: list[ToolDefinition] | None = None,
        handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        self._definitions = {item.tool_id: item for item in definitions or []}
        self._handlers = dict(handlers or {})
        self.descriptor = ServiceDescriptor(
            service_name="tools",
            implementation_id="registered_tool_service",
            maturity="simple",
            capabilities=sorted(self._definitions.keys()),
        )

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        handler = self._handlers.get(request.tool_id)
        if handler is None:
            return ToolExecutionResponse(
                status="unavailable",
                metadata={"reason": f"tool '{request.tool_id}' is not registered"},
            )
        return handler(request)

    def list_tools(self) -> list[ToolDefinition]:
        return [self._definitions[key] for key in sorted(self._definitions)]

    @classmethod
    def with_defaults(cls, *, web_search_client: Any | None = None) -> "RegisteredToolService":
        """Create the default toolbox with built-in tools."""
        definitions = [
            ToolDefinition(
                tool_id="web_search",
                name="Web Search",
                description="Search the public web for up-to-date information.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                        "topic": {"type": "string", "default": "general"},
                        "include_raw_content": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
                metadata={"provider": "tavily"},
            )
        ]

        def web_search_handler(request: ToolExecutionRequest) -> ToolExecutionResponse:
            query = str(request.arguments.get("query", "")).strip()
            if not query:
                return ToolExecutionResponse(
                    status="error",
                    metadata={"reason": "query is required"},
                )

            client = web_search_client
            if client is None:
                try:
                    client = TavilyWebSearchClient()
                except (ImportError, ValueError) as exc:
                    return ToolExecutionResponse(
                        status="unavailable",
                        metadata={"reason": str(exc)},
                    )

            try:
                result = client.search(
                    query=query,
                    max_results=int(request.arguments.get("max_results", 5)),
                    topic=str(request.arguments.get("topic", "general")),
                    include_raw_content=bool(request.arguments.get("include_raw_content", False)),
                )
            except Exception as exc:
                return ToolExecutionResponse(
                    status="error",
                    metadata={"reason": str(exc), "tool_id": "web_search"},
                )
            return ToolExecutionResponse(
                status="succeeded",
                output=result,
                metadata={"tool_id": "web_search"},
            )

        return cls(
            definitions=definitions,
            handlers={"web_search": web_search_handler},
        )

