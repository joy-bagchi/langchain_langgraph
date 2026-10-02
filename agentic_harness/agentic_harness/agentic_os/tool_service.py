"""Tool service contract and default registered tool implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from agentic_harness.shared.services import ServiceDescriptor
from agentic_harness.ibkr_data_reader import (
    IBKR_ACTIONS,
    IBKRAuthorizationRequired,
    IBKRCredentialStore,
    IBKRCredentials,
    IBKRDataProvider,
    IBKRScopeRejected,
    IBKRStorageUnavailable,
    ProviderResult,
    SecretManagerIBKRCredentialStore,
    _require_exact_read_scope,
    normalize_provider_result,
    validate_action_arguments,
)
from agentic_harness.notifications import (
    HarnessNotificationService,
    NotificationService,
    WorkflowNotification,
)


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
        ibkr_data_reader_provider: IBKRDataProvider | None = None,
        ibkr_credential_store: IBKRCredentialStore | None = None,
        notification_service: NotificationService | None = None,
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
        ]

        def web_search_handler(request: ToolExecutionRequest) -> ToolExecutionResponse:
            retrieved_at = datetime.now(timezone.utc).isoformat()
            query = str(request.arguments.get("query", "")).strip()
            if not query:
                return ToolExecutionResponse(
                    status="error",
                    metadata={"reason": "query is required", "query": query,
                              "retrieved_at": retrieved_at},
                )

            client = web_search_client
            if client is None:
                try:
                    client = TavilyWebSearchClient()
                except (ImportError, ValueError) as exc:
                    return ToolExecutionResponse(
                        status="unavailable",
                        metadata={"reason": str(exc), "query": query,
                                  "retrieved_at": retrieved_at},
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
                    metadata={"reason": str(exc), "tool_id": "web_search",
                              "query": query, "retrieved_at": retrieved_at},
                )
            sources = []
            if isinstance(result, dict):
                for item in result.get("results", []):
                    if isinstance(item, dict):
                        sources.append({key: item[key] for key in
                                        ("url", "title", "published_date", "source_timestamp", "quote_timestamp")
                                        if item.get(key) is not None})
            return ToolExecutionResponse(
                status="succeeded",
                output=result,
                metadata={"tool_id": "web_search", "query": query,
                          "retrieved_at": retrieved_at, "sources": sources},
            )

        notify = notification_service or HarnessNotificationService()

        action_schemas = {
            "get_symbol_daily_data": {
                "type": "object", "additionalProperties": False,
                "properties": {"symbol": {"type": "string"}, "trading_date": {"type": "string"}},
                "required": ["symbol", "trading_date"],
            },
            "list_option_contracts": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "symbol": {"type": "string"}, "expiry": {"type": ["string", "null"]},
                    "pagination": {"type": "object", "additionalProperties": False,
                                   "properties": {"cursor": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
                },
                "required": ["symbol"],
            },
            "get_option_data": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "exact_contract_id": {"type": ["string", "integer"]},
                    "symbol": {"type": "string"}, "right": {"type": "string", "enum": ["C", "P", "CALL", "PUT"]},
                    "strike": {"type": "number"}, "expiry": {"type": "string"},
                },
                "oneOf": [
                    {"required": ["exact_contract_id"],
                     "not": {"anyOf": [{"required": [field]} for field in ("symbol", "right", "strike", "expiry")]}},
                    {"required": ["symbol", "right", "strike", "expiry"],
                     "not": {"required": ["exact_contract_id"]}},
                ],
            },
        }
        definitions.extend(
            ToolDefinition(
                tool_id=action,
                name=action,
                description={
                    "get_symbol_daily_data": "Read daily price and volume fields for the requested symbol and trading date.",
                    "list_option_contracts": "List exact option contract identifiers, rights, strikes, and expiries for the requested symbol.",
                    "get_option_data": "Read market fields for one exact option contract. No alternate strike or expiry is selected.",
                }[action],
                input_schema=action_schemas[action],
                metadata={"provider": "ibkr_data_reader", "tool_type": "ibkr_data_reader", "read_only": True},
            )
            for action in IBKR_ACTIONS
        )

        def ibkr_action_handler(request: ToolExecutionRequest) -> ToolExecutionResponse:
            action = request.tool_id
            try:
                arguments = validate_action_arguments(action, dict(request.arguments))
            except (TypeError, ValueError):
                return ToolExecutionResponse(status="error", metadata={"reason": "invalid_action_arguments", "tool_id": action})

            def authorization_required(reason: str) -> ToolExecutionResponse:
                notification = WorkflowNotification(
                    notification_type="ibkr_authorization_required",
                    message="IBKR read-only consent is required before this workflow can continue.",
                    run_id=request.metadata.get("run_id"),
                    workflow_id=request.metadata.get("workflow_id"),
                    step_id=request.metadata.get("step_id"),
                    metadata={"tool_id": action, "status": "authorization_required"},
                )
                notify.notify(notification)
                return ToolExecutionResponse(
                    status="authorization_required",
                    output={"status": "authorization_required", "tool_id": action},
                    metadata={"reason": reason, "notification_type": notification.notification_type,
                              "tool_id": action, "fresh_market_data_required": True},
                )

            try:
                store = ibkr_credential_store or SecretManagerIBKRCredentialStore()
                credentials = store.load()
                if credentials is None:
                    return authorization_required("read_only_consent_required")
                _require_exact_read_scope(credentials)
            except IBKRScopeRejected:
                return ToolExecutionResponse(status="scope_rejected", metadata={"reason": "read_only_scope_required", "tool_id": action})
            except IBKRStorageUnavailable:
                return ToolExecutionResponse(status="unavailable", metadata={"reason": "protected_credential_storage_unavailable", "tool_id": action})
            if ibkr_data_reader_provider is None:
                return ToolExecutionResponse(status="unavailable", metadata={"reason": "authenticated_provider_mapping_unavailable", "tool_id": action})

            try:
                refreshed = ibkr_data_reader_provider.refresh_credentials(credentials)
                _require_exact_read_scope(refreshed)
                if refreshed.tokens != credentials.tokens or refreshed.client_info != credentials.client_info:
                    store.persist(refreshed)
                provider_result = ibkr_data_reader_provider.invoke(action, arguments, refreshed)
                if not isinstance(provider_result, ProviderResult):
                    raise TypeError("provider result contract mismatch")
                final_credentials = provider_result.rotated_credentials
                if final_credentials is not None:
                    _require_exact_read_scope(final_credentials)
                    store.persist(final_credentials)
                output = normalize_provider_result(action, arguments, provider_result.payload)
            except IBKRAuthorizationRequired:
                return authorization_required("read_only_consent_required")
            except IBKRScopeRejected:
                return ToolExecutionResponse(status="scope_rejected", metadata={"reason": "read_only_scope_required", "tool_id": action})
            except IBKRStorageUnavailable:
                return ToolExecutionResponse(status="unavailable", metadata={"reason": "protected_credential_storage_unavailable", "tool_id": action})
            except Exception:
                # Provider exception strings may contain protected provider data.
                return ToolExecutionResponse(status="unavailable", metadata={"reason": "provider_unavailable", "tool_id": action})
            return ToolExecutionResponse(status="succeeded", output=output, metadata={"tool_id": action})

        return cls(
            definitions=definitions,
            handlers={
                "web_search": web_search_handler,
                **{action: ibkr_action_handler for action in IBKR_ACTIONS},
            },
        )

