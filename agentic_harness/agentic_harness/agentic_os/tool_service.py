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
    def with_defaults(
        cls,
        *,
        web_search_client: Any | None = None,
        ibkr_data_pipe: Any | None = None,
    ) -> "RegisteredToolService":
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
            ),
            ToolDefinition(
                tool_id="ibkr_data_pipeline",
                name="IBKR Data Pipeline",
                description=(
                    "Fetch Interactive Brokers market data for an underlying and a selected "
                    "option chain, including prices, volume, open interest, and Greeks."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "default": "fetch_market_snapshot",
                            "enum": ["fetch_market_snapshot", "fetch_vol_regime_snapshot"],
                        },
                        "symbol": {"type": "string", "default": "SPY"},
                        "host": {"type": "string", "default": "127.0.0.1"},
                        "port": {"type": "integer", "default": 4001},
                        "client_id": {"type": "integer", "default": 73},
                        "market_data_type": {"type": "integer", "default": 1},
                        "exchange": {"type": "string", "default": "SMART"},
                        "option_exchange": {"type": "string", "default": "SMART"},
                        "currency": {"type": "string", "default": "USD"},
                        "rights": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["C", "P"],
                        },
                        "expiry_count": {"type": "integer", "default": 2},
                        "strike_count": {"type": "integer", "default": 8},
                        "expirations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "strikes": {
                            "type": "array",
                            "items": {"type": "number"},
                            "default": [],
                        },
                        "min_days_to_expiry": {"type": "integer", "default": 0},
                        "history_days": {"type": "integer", "default": 30},
                        "regime_symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["SPY", "VIX", "VVIX", "VIX9D", "VIX3M"],
                        },
                        "index_exchange": {"type": "string", "default": "CBOE"},
                    },
                    "required": ["operation"],
                },
                metadata={"provider": "interactive_brokers", "default_port": 4001},
            ),
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

        def ibkr_data_pipeline_handler(request: ToolExecutionRequest) -> ToolExecutionResponse:
            operation = str(request.arguments.get("operation", "fetch_market_snapshot")).strip().lower()
            if operation not in {"fetch_market_snapshot", "fetch_vol_regime_snapshot"}:
                return ToolExecutionResponse(
                    status="error",
                    metadata={
                        "reason": f"unsupported operation '{operation}'",
                        "tool_id": "ibkr_data_pipeline",
                    },
                )

            try:
                from agentic_vol_regime_app.data.ibkr_client import (
                    IBKRConnectionConfig,
                    IBKRDataPipe,
                    IBKROptionChainRequest,
                    IBKRVolRegimeSnapshotRequest,
                )
            except ImportError as exc:
                return ToolExecutionResponse(
                    status="unavailable",
                    metadata={
                        "reason": (
                            "agentic_vol_regime_app IBKR integration is not available. "
                            f"{exc}"
                        ),
                        "tool_id": "ibkr_data_pipeline",
                    },
                )

            arguments = dict(request.arguments)
            pipe = ibkr_data_pipe
            if pipe is None:
                pipe = IBKRDataPipe(
                    connection=IBKRConnectionConfig(
                        host=str(arguments.get("host", "127.0.0.1")),
                        port=int(arguments.get("port", 4001)),
                        client_id=int(arguments.get("client_id", 73)),
                        market_data_type=int(arguments.get("market_data_type", 1)),
                    )
                )

            try:
                if operation == "fetch_vol_regime_snapshot":
                    observation = pipe.fetch_vol_regime_snapshot(
                        IBKRVolRegimeSnapshotRequest.from_payload(arguments)
                    )
                else:
                    observation = pipe.fetch_market_snapshot(
                        IBKROptionChainRequest(
                            symbol=str(arguments.get("symbol", "SPY")),
                            exchange=str(arguments.get("exchange", "SMART")),
                            currency=str(arguments.get("currency", "USD")),
                            option_exchange=str(arguments.get("option_exchange", "SMART")),
                            rights=tuple(
                                str(item).upper() for item in arguments.get("rights", ["C", "P"])
                            ),
                            expiry_count=int(arguments.get("expiry_count", 2)),
                            strike_count=int(arguments.get("strike_count", 8)),
                            expirations=tuple(
                                str(item) for item in arguments.get("expirations", [])
                            ),
                            strikes=tuple(
                                float(item) for item in arguments.get("strikes", [])
                            ),
                            min_days_to_expiry=int(arguments.get("min_days_to_expiry", 0)),
                        )
                    )
            except Exception as exc:
                return ToolExecutionResponse(
                    status="error",
                    metadata={"reason": str(exc), "tool_id": "ibkr_data_pipeline"},
                )
            output = observation.to_dict() if hasattr(observation, "to_dict") else observation
            return ToolExecutionResponse(
                status="succeeded",
                output=output,
                metadata={
                    "tool_id": "ibkr_data_pipeline",
                    "operation": operation,
                    "symbol": str(arguments.get("symbol", "SPY")),
                    "port": int(arguments.get("port", 4001)),
                },
            )

        return cls(
            definitions=definitions,
            handlers={
                "web_search": web_search_handler,
                "ibkr_data_pipeline": ibkr_data_pipeline_handler,
            },
        )

