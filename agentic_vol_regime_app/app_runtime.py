"""Runtime wrapper for running the volatility regime app on agentic_harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from agentic_vol_regime_app._bootstrap import ensure_repo_imports

ensure_repo_imports()

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.contracts import AgentRuntimeProfile, ContextPolicy, MemoryLifecyclePolicy
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.runtime import resume_workflow, start_workflow

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.executors import build_executor_registry


def default_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_orchestrator.yaml"


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
    )
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
    result["agent"] = {
        "agent_id": agent_definition.agent_id,
        "name": agent_definition.name,
        "role": agent_definition.role,
        "workflow_path": agent_definition.workflow_path,
        "memory_service_type": agent_definition.memory_service_type,
        "runtime_profile": agent_definition.runtime_profile,
    }
    return result


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
    return resume_workflow(
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
