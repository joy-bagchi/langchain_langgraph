"""Runtime wrapper for running the volatility regime app on agentic_harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentic_vol_regime_app._bootstrap import ensure_repo_imports

ensure_repo_imports()

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.contracts import AgentRuntimeProfile, ContextPolicy, MemoryLifecyclePolicy, MemoryQuery
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.runtime import resume_workflow, run_agent_workflow, start_workflow
from agentic_harness.stores import FilesystemMemoryStore

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.executors import build_executor_registry


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


def default_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_orchestrator.yaml"


def default_ml_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_ml_orchestrator.yaml"


def default_hmm_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_hmm_orchestrator.yaml"


def default_ibkr_agent_path() -> Path:
    return AppPaths.default().agents_dir / "ibkr_market_data_agent.yaml"


def _resolve_memory_namespace(agent_path: str | Path | None = None) -> str:
    resolved_agent_path = Path(agent_path or default_agent_path()).resolve()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    return agent_definition.memory_namespace or f"{agent_definition.agent_id}_memory"


def _resolve_runtime_profile(profile_id: str) -> AgentRuntimeProfile:
    lowered = (profile_id or "default").strip().lower()
    if lowered == "durable_research":
        return AgentRuntimeProfile(
            profile_id=lowered,
            context_policy=ContextPolicy(
                max_recent_history=5,
                max_memory_hits=5,
                max_working_notes_chars=1200,
                token_budget=2200,
            ),
            memory_policy=MemoryLifecyclePolicy(
                max_ephemeral_records=250,
                max_durable_records=20000,
            ),
            suspension_threshold_seconds=900,
        )
    if lowered == "short_task":
        return AgentRuntimeProfile(
            profile_id=lowered,
            context_policy=ContextPolicy(
                max_recent_history=2,
                max_memory_hits=2,
                max_working_notes_chars=300,
                token_budget=900,
            ),
            memory_policy=MemoryLifecyclePolicy(
                max_ephemeral_records=25,
                max_durable_records=500,
            ),
            suspension_threshold_seconds=120,
        )
    return AgentRuntimeProfile(profile_id="default")


def run_daily_regime_agent(
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
    ibkr_data_pipe=None,
) -> dict[str, Any]:
    """Run the default daily volatility-regime workflow."""
    resolved_agent_path = Path(agent_path or default_agent_path()).resolve()
    app_paths = AppPaths.default()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    runtime_profile = _resolve_runtime_profile(agent_definition.runtime_profile)
    services = build_platform_services(
        storage_root=storage_root,
        memory_service_type=agent_definition.memory_service_type,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
        ibkr_data_pipe=ibkr_data_pipe,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:run_daily_regime_agent",
        run_type="chain",
        inputs={
            "agent_id": agent_definition.agent_id,
            "agent_name": agent_definition.name,
            "workflow_path": agent_definition.workflow_path,
            "input_payload": _trace_safe_payload(input_payload),
        },
        tags=["agentic_vol_regime_app", "daily_regime_agent"],
        metadata={
            "application": "agentic_vol_regime_app",
            "agent_id": agent_definition.agent_id,
            "agent_role": agent_definition.role,
            "runtime_profile": runtime_profile.profile_id,
        },
    ) as app_span:
        result = start_workflow(
            agent_definition.workflow_path,
            input_payload,
            storage_root=storage_root,
            services=services,
            executors=build_executor_registry(app_paths=app_paths, services=services),
            memory_service_type=agent_definition.memory_service_type,
            database_url=database_url,
            initial_state_overrides={
                "agent_id": agent_definition.agent_id,
                "agent_name": agent_definition.name,
                "agent_role": agent_definition.role,
                "agent_metadata": dict(agent_definition.metadata),
                "memory_namespace": agent_definition.memory_namespace or f"{agent_definition.agent_id}_memory",
                "invocation_id": str(uuid4()),
                "runtime_profile": runtime_profile.profile_id,
                "context_policy": runtime_profile.context_policy.to_dict(),
                "memory_policy": runtime_profile.memory_policy.to_dict(),
                "allowed_tools": list(agent_definition.allowed_tools),
            },
            langsmith_tracing=langsmith_tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
            langsmith_workspace_id=langsmith_workspace_id,
        )
        if hasattr(app_span, "end"):
            app_span.end(
                outputs={
                    "status": result.get("status"),
                    "run_id": result.get("run_id"),
                    "workflow_id": result.get("workflow_id"),
                    "current_step": result.get("current_step"),
                    "pending_review": _trace_safe_payload(result.get("pending_review")),
                    "last_error": result.get("last_error"),
                }
            )
    result["agent"] = {
        "agent_id": agent_definition.agent_id,
        "name": agent_definition.name,
        "role": agent_definition.role,
        "workflow_path": agent_definition.workflow_path,
        "memory_service_type": agent_definition.memory_service_type,
        "runtime_profile": agent_definition.runtime_profile,
    }
    return result


def load_latest_live_daily_observation(
    *,
    agent_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
) -> dict[str, Any] | None:
    """Return the latest live daily observation snapshot persisted in memory."""
    namespace = _resolve_memory_namespace(agent_path)
    store_root = Path(storage_root or Path.cwd() / ".workflow_memory")
    memory_store = FilesystemMemoryStore(store_root, database_url=database_url)
    matches = memory_store.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=1,
            memory_types=["live_observation_snapshot"],
            structured_filters={"source_kind": "live_ibkr"},
        )
    )
    if not matches:
        return None
    structured_payload = matches[0].record.structured_payload
    observation = structured_payload.get("observation")
    if isinstance(observation, dict):
        return observation
    return None


def load_recent_hmm_state_history(
    *,
    agent_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return recent HMM state snapshots persisted through harness memory."""
    namespace = _resolve_memory_namespace(agent_path)
    store_root = Path(storage_root or Path.cwd() / ".workflow_memory")
    memory_store = FilesystemMemoryStore(store_root, database_url=database_url)
    matches = memory_store.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=max(1, int(limit)),
            memory_types=["hmm_state_snapshot"],
            structured_filters={"source_kind": "hmm_gaussian", "is_trained": True},
        )
    )
    history: list[dict[str, Any]] = []
    for match in matches:
        payload = dict(match.record.structured_payload)
        payload["record_id"] = match.record.record_id
        payload["created_at"] = match.record.created_at
        payload["memory_type"] = match.record.memory_type
        history.append(payload)
    return history


def resume_daily_regime_run(
    *,
    run_id: str,
    decision: str | None = None,
    notes: str | None = None,
    agent_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Resume a review-gated daily volatility-regime run."""
    resolved_agent_path = Path(agent_path or default_agent_path()).resolve()
    app_paths = AppPaths.default()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    services = build_platform_services(
        storage_root=storage_root,
        memory_service_type=agent_definition.memory_service_type,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:resume_daily_regime_run",
        run_type="chain",
        inputs={
            "run_id": run_id,
            "decision": decision,
            "notes": _trace_safe_payload(notes),
            "agent_id": agent_definition.agent_id,
        },
        tags=["agentic_vol_regime_app", "daily_regime_resume"],
        metadata={
            "application": "agentic_vol_regime_app",
            "agent_id": agent_definition.agent_id,
            "agent_role": agent_definition.role,
        },
    ) as app_span:
        result = resume_workflow(
            run_id,
            storage_root=storage_root,
            decision=decision,
            notes=notes,
            services=services,
            executors=build_executor_registry(app_paths=app_paths, services=services),
            memory_service_type=agent_definition.memory_service_type,
            database_url=database_url,
            langsmith_tracing=langsmith_tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
            langsmith_workspace_id=langsmith_workspace_id,
        )
        if hasattr(app_span, "end"):
            app_span.end(
                outputs={
                    "status": result.get("status"),
                    "run_id": result.get("run_id"),
                    "workflow_id": result.get("workflow_id"),
                    "current_step": result.get("current_step"),
                    "pending_review": _trace_safe_payload(result.get("pending_review")),
                    "last_error": result.get("last_error"),
                }
            )
    return result


def run_ibkr_market_data_agent(
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
    """Run the IBKR tool-backed market data agent through the harness runtime."""
    resolved_agent_path = Path(agent_path or default_ibkr_agent_path()).resolve()
    services = build_platform_services(
        storage_root=storage_root,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    with services.observability.trace_span(
        "agentic_vol_regime_app:run_ibkr_market_data_agent",
        run_type="chain",
        inputs={
            "agent_id": agent_definition.agent_id,
            "workflow_path": agent_definition.workflow_path,
            "input_payload": _trace_safe_payload(input_payload),
        },
        tags=["agentic_vol_regime_app", "ibkr_market_data_agent"],
        metadata={
            "application": "agentic_vol_regime_app",
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
                    "current_step": result.get("current_step"),
                    "pending_review": _trace_safe_payload(result.get("pending_review")),
                    "last_error": result.get("last_error"),
                }
            )
    return result
