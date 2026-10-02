#!/usr/bin/env python3
"""Small Streamable HTTP MCP client with browser-based OAuth.

The MCP SDK performs OAuth discovery, registration, PKCE, callback validation,
and token refresh. Credentials live only for this process; this script does not
write tokens or client credentials to disk.

Examples:
    python scripts/mcp_oauth_client.py https://example.com/mcp
    python scripts/mcp_oauth_client.py https://example.com/mcp \
        --call get_status --arguments '{"detail": true}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import re
import sys
import threading
import webbrowser
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken


CALLBACK_TIMEOUT_SECONDS = 300
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
SENSITIVE_KEY_RE = re.compile(
    r"access[_-]?token|refresh[_-]?token|token|secret|credential|password|"
    r"authorization|api[_-]?key|auth[_-]?code|state",
    re.IGNORECASE,
)


class EphemeralOAuthStorage(TokenStorage):
    """Minimal SDK storage adapter that keeps OAuth values in process memory."""

    def __init__(self) -> None:
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info

    def secret_values(self) -> tuple[str, ...]:
        values: list[str] = []
        if self.tokens is not None:
            values.extend((self.tokens.access_token, self.tokens.refresh_token or ""))
        if self.client_info is not None:
            values.extend((self.client_info.client_secret or "",))
        return tuple(value for value in values if value)


class LoopbackCallback:
    """Short-lived localhost callback listener for the SDK's OAuth flow."""

    def __init__(self, timeout: int = CALLBACK_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self.result: queue.Queue[tuple[str, str | None, str | None]] = queue.Queue(maxsize=1)
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "LoopbackCallback":
        result_queue = self.result

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                parsed = urlsplit(self.path)
                if parsed.path != "/callback":
                    self.send_error(404)
                    return

                params = parse_qs(parsed.query, keep_blank_values=True)
                code = _single_value(params, "code")
                state = _single_value(params, "state")
                error = _single_value(params, "error")
                if code and state and not error:
                    _offer_callback(result_queue, ("success", code, state))
                    status, body = 200, b"Authorization received. You may close this window."
                elif error:
                    _offer_callback(result_queue, ("error", None, state))
                    status, body = 400, b"Authorization was not completed. You may close this window."
                else:
                    status, body = 400, b"Invalid authorization callback. You may close this window."

                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                # The default access log includes the callback path and query.
                return

        class SilentCallbackServer(ThreadingHTTPServer):
            def handle_error(self, _request: Any, _client_address: Any) -> None:
                # Do not let server tracebacks print callback request details.
                return

        self.server = SilentCallbackServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def redirect_uri(self) -> str:
        if self.server is None:
            raise RuntimeError("Callback listener is not running")
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/callback"

    async def wait_for_callback(self) -> tuple[str, str | None]:
        try:
            kind, code, state = await asyncio.to_thread(self.result.get, True, self.timeout)
        except queue.Empty as exc:
            raise RuntimeError("Timed out waiting for the OAuth callback") from exc
        if kind != "success" or code is None:
            raise RuntimeError("Authorization was not completed")
        return code, state

    def __exit__(self, *_exc: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)


def _single_value(params: Mapping[str, list[str]], key: str) -> str | None:
    values = params.get(key, [])
    return values[0] if len(values) == 1 and values[0] else None


def _offer_callback(
    destination: queue.Queue[tuple[str, str | None, str | None]],
    value: tuple[str, str | None, str | None],
) -> None:
    try:
        destination.put_nowait(value)
    except queue.Full:
        pass


def _safe_authorization_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not (
        parsed.username or parsed.password
    )


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if SENSITIVE_KEY_RE.search(str(key)) else _redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return JWT_RE.sub("[REDACTED]", BEARER_RE.sub("Bearer [REDACTED]", value))
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump(mode="json"), secrets)
    return value


def _parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("arguments must be a JSON object")
    return value


async def run_client(args: argparse.Namespace) -> None:
    storage = EphemeralOAuthStorage()
    with LoopbackCallback() as callback:
        async def open_login_page(authorization_url: str) -> None:
            if not _safe_authorization_url(authorization_url):
                raise RuntimeError("The MCP server supplied an invalid authorization URL")
            if not webbrowser.open(authorization_url, new=1, autoraise=True):
                raise RuntimeError("Could not open the authorization page in a browser")
            print("An authorization page has opened in your browser. Complete sign-in there.")

        oauth = OAuthClientProvider(
            server_url=args.server_url,
            client_metadata=OAuthClientMetadata(
                redirect_uris=[callback.redirect_uri],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
                client_name="Local MCP OAuth CLI",
            ),
            storage=storage,
            redirect_handler=open_login_page,
            callback_handler=callback.wait_for_callback,
            timeout=CALLBACK_TIMEOUT_SECONDS,
        )

        async with httpx.AsyncClient(auth=oauth) as http_client:
            async with streamable_http_client(args.server_url, http_client=http_client) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    if args.call:
                        result = await session.call_tool(args.call, args.arguments)
                        safe_result = _redact(result, storage.secret_values())
                        if hasattr(safe_result, "model_dump"):
                            safe_result = safe_result.model_dump(mode="json")
                        print(json.dumps(safe_result, ensure_ascii=False, default=str, indent=2))
                    else:
                        result = await session.list_tools()
                        print("Available tools:")
                        secrets = storage.secret_values()
                        for tool in result.tools:
                            name = _redact(tool.name, secrets)
                            description = _redact(tool.description or "", secrets)
                            suffix = f" — {description}" if description else ""
                            print(f"- {name}{suffix}")


def main() -> int:
    # Library diagnostics can include request URLs or provider response text.
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Connect to an MCP server over Streamable HTTP with browser OAuth.")
    parser.add_argument("server_url", help="MCP Streamable HTTP endpoint (http or https)")
    parser.add_argument("--call", metavar="TOOL", help="call one tool after connecting")
    parser.add_argument(
        "--arguments",
        type=_parse_arguments,
        default={},
        help="JSON object of tool arguments (used with --call)",
    )
    args = parser.parse_args()
    if args.arguments and not args.call:
        parser.error("--arguments requires --call")

    try:
        asyncio.run(run_client(args))
    except Exception as exc:
        # Avoid printing provider exception messages, which may contain URLs,
        # response bodies, or other authentication details.
        print(f"MCP client failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
