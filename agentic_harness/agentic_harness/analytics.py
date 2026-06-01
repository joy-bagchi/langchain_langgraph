"""Amplitude analytics integration for the agentic harness CLI."""

from __future__ import annotations

import os
from typing import Any

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("AMPLITUDE_API_KEY")
    if not api_key:
        return None
    try:
        from amplitude import Amplitude
        _client = Amplitude(api_key)
    except Exception:
        pass
    return _client


def track(event_name: str, properties: dict[str, Any] | None = None) -> None:
    """Track an event. No-ops silently if Amplitude is not configured."""
    client = _get_client()
    if client is None:
        return
    try:
        from amplitude import BaseEvent
        client.track(BaseEvent(
            event_type=event_name,
            user_id="cli",
            event_properties=properties or {},
        ))
    except Exception:
        pass


def shutdown() -> None:
    """Flush and close the Amplitude client."""
    client = _get_client()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        pass
