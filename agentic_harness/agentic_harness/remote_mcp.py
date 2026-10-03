"""Reusable, model-hidden client capability for remote MCP servers.

Only trusted provider adapters should use this module. It deliberately offers
initialization and mapped tool invocation, never the provider's raw catalog.
OAuth policy and credential storage are injected by the Harness host.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

from agentic_harness.ibkr_data_reader import READ_ONLY_SCOPE


class RemoteMCPScopeError(ValueError):
    """A remote MCP credential exceeds its configured exact scope ceiling."""


def require_exact_scope(scope: str | None, *, expected_scope: str = READ_ONLY_SCOPE) -> None:
    """Fail closed unless an OAuth scope set contains exactly one approved scope."""
    if scope is None or set(scope.split()) != {expected_scope}:
        raise RemoteMCPScopeError(f"Remote MCP grant must contain exactly {expected_scope}")


@dataclass(frozen=True, slots=True)
class RemoteMCPProfile:
    server_url: str
    scope: str = READ_ONLY_SCOPE

    def __post_init__(self) -> None:
        require_exact_scope(self.scope, expected_scope=self.scope)


class RemoteMCPConnection:
    """Trusted-only view of an initialized remote MCP session."""

    def __init__(self, session: Any, tool_bindings: Mapping[str, str]) -> None:
        self._session = session
        self._tool_bindings = dict(tool_bindings)

    async def invoke_mapped(self, action: str, arguments: dict[str, Any]) -> Any:
        """Invoke only a reviewed internal action-to-provider-tool binding."""
        provider_tool = self._tool_bindings.get(action)
        if not provider_tool:
            raise LookupError("Remote MCP action has no reviewed provider mapping")
        return await self._session.call_tool(provider_tool, arguments)


class RemoteMCPClient:
    """MCP SDK transport wrapper with no catalog surface exposed to agents."""

    def __init__(
        self,
        profile: RemoteMCPProfile,
        *,
        tool_bindings: Mapping[str, str] | None = None,
        http_timeout: float = 30.0,
    ) -> None:
        self.profile = profile
        self._tool_bindings = dict(tool_bindings or {})
        self._http_timeout = http_timeout

    @asynccontextmanager
    async def connect(self, auth: Any, *, http_client: Any | None = None) -> AsyncIterator[RemoteMCPConnection]:
        """Initialize an SDK ClientSession over authenticated Streamable HTTP."""
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx

        if http_client is not None:
            async with streamable_http_client(
                self.profile.server_url,
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    # The caller receives only this constrained wrapper; there
                    # is no list_tools/list_resources or raw-session attribute.
                    yield RemoteMCPConnection(session, self._tool_bindings)
            return

        async with httpx.AsyncClient(
            auth=auth,
            timeout=httpx.Timeout(self._http_timeout, read=self._http_timeout + 15),
        ) as owned_client:
            async with streamable_http_client(
                self.profile.server_url,
                http_client=owned_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield RemoteMCPConnection(session, self._tool_bindings)


__all__ = ["RemoteMCPClient", "RemoteMCPConnection", "RemoteMCPProfile", "RemoteMCPScopeError", "require_exact_scope"]
