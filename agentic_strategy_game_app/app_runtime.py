"""Runtime wrappers for strategy-game agents that sit on top of agentic_harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_strategy_game_app._bootstrap import ensure_repo_imports

ensure_repo_imports()

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.llm import build_model_callable, resolve_llm_config
from agentic_harness.runtime import run_agent_workflow


def _trace_safe_payload(value: Any, *, max_string_chars: int = 1000, max_items: int = 10, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else f"{value[:max_string_chars]}...<truncated>"
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        result = {
            str(key): _trace_safe_payload(item, max_string_chars=max_string_chars, max_items=max_items, depth=depth + 1)
            for key, item in items
        }
        if len(value) > max_items:
            result["__truncated_keys__"] = len(value) - max_items
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _trace_safe_payload(item, max_string_chars=max_string_chars, max_items=max_items, depth=depth + 1)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            result.append(f"...<truncated {len(value) - max_items} items>")
        return result
    try:
        return _trace_safe_payload(
            json.loads(json.dumps(value, default=str)),
            max_string_chars=max_string_chars,
            max_items=max_items,
            depth=depth + 1,
        )
    except Exception:
        rendered = str(value)
        return rendered if len(rendered) <= max_string_chars else f"{rendered[:max_string_chars]}...<truncated>"


def default_vc_agent_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "agents" / "vc_investor_agent.yaml"


def run_vc_investor_agent(
    *,
    input_payload: dict[str, Any],
    agent_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    resolved_agent_path = Path(agent_path or default_vc_agent_path()).resolve()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    llm_config = resolve_llm_config(
        provider=agent_definition.llm_provider,
        model=agent_definition.model,
        temperature=agent_definition.temperature,
    )
    services = build_platform_services(
        storage_root=storage_root,
        model_callable=build_model_callable(llm_config),
        memory_service_type=agent_definition.memory_service_type,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_strategy_game_app:run_vc_investor_agent",
        run_type="chain",
        inputs={
            "agent_id": agent_definition.agent_id,
            "workflow_path": agent_definition.workflow_path,
            "input_payload": _trace_safe_payload(input_payload),
        },
        tags=["agentic_strategy_game_app", "vc_investor_agent"],
        metadata={
            "application": "agentic_strategy_game_app",
            "agent_id": agent_definition.agent_id,
            "agent_role": agent_definition.role,
        },
    ) as app_span:
        result = run_agent_workflow(
            resolved_agent_path,
            input_payload,
            storage_root=storage_root,
            database_url=database_url,
            langsmith_tracing=langsmith_tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
            langsmith_workspace_id=langsmith_workspace_id,
            services=services,
        )
        if hasattr(app_span, "end"):
            app_span.end(
                outputs={
                    "status": result.get("status"),
                    "run_id": result.get("run_id"),
                    "workflow_id": result.get("workflow_id"),
                    "named_outputs": _trace_safe_payload(result.get("named_outputs", {})),
                    "last_error": result.get("last_error"),
                }
            )
    return result
