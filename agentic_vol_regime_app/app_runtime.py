"""Runtime wrapper for running the volatility regime app on agentic_harness."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from agentic_vol_regime_app._bootstrap import ensure_repo_imports

ensure_repo_imports()

from agentic_harness.agentic_os.platform import build_platform_services
from agentic_harness.contracts import AgentRuntimeProfile, ContextPolicy, MemoryLifecyclePolicy, MemoryQuery
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.runtime import resume_workflow, run_agent_workflow, start_workflow
from agentic_harness.stores import FilesystemMemoryStore

from agentic_vol_regime_app.config import AppPaths
from agentic_vol_regime_app.executors import build_executor_registry
from agentic_vol_regime_app.pomdp.hmm_belief import load_hmm_config
from agentic_vol_regime_app.reports.daily_report import resolve_daily_report_path


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


def default_hmm_v2_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_hmm_v2_orchestrator.yaml"


def default_hmm_v3_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_hmm_v3_orchestrator.yaml"


def default_ibkr_agent_path() -> Path:
    return AppPaths.default().agents_dir / "ibkr_market_data_agent.yaml"


def _resolve_report_model_identity(agent_path: str | Path | None = None) -> tuple[str, str]:
    resolved_agent_path = Path(agent_path or default_agent_path()).resolve()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    belief_engine = str(dict(agent_definition.metadata).get("belief_engine", "heuristic")).strip().lower()
    if belief_engine == "ml_linear_regression":
        return ("LinearRegimeBeliefModel", "linear_regression_regime_v1")
    if belief_engine == "hmm_gaussian_v3":
        return ("HMMBeliefAgent", "hmm_gaussian_v3")
    if belief_engine == "hmm_gaussian_v2":
        return ("HMMBeliefAgent", "hmm_gaussian_v2")
    if belief_engine.startswith("hmm_gaussian"):
        return ("HMMBeliefAgent", "hmm_gaussian_v1")
    return ("HeuristicBeliefModel", "belief_model_v1")


def _find_existing_daily_report_path(
    *,
    report_root: str | Path,
    as_of_date: str,
    report_model_name: str,
    report_model_version: str,
) -> Path | None:
    reports_dir = Path(report_root).resolve() / "daily"
    exact_path = resolve_daily_report_path(
        report_root=Path(report_root).resolve(),
        as_of=as_of_date,
        report_model_name=report_model_name,
        report_model_version=report_model_version,
    )
    if exact_path.exists():
        return exact_path
    legacy_path = resolve_daily_report_path(
        report_root=Path(report_root).resolve(),
        as_of=as_of_date,
        report_model_name=report_model_name,
        report_model_version=None,
    )
    if legacy_path.exists():
        return legacy_path
    if not reports_dir.exists():
        return None
    pattern = f"daily_regime_report_{as_of_date[:10]}_*.md"
    candidates = sorted(
        reports_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            markdown = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if report_model_name in markdown and report_model_version in markdown:
            return candidate
    return None


def load_historical_belief_report(
    *,
    as_of_date: str,
    agent_path: str | Path | None = None,
    report_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
) -> dict[str, Any] | None:
    report_model_name, report_model_version = _resolve_report_model_identity(agent_path)
    namespace = _resolve_memory_namespace(agent_path)
    store_root = Path(storage_root or Path.cwd() / ".workflow_memory")
    memory_store = FilesystemMemoryStore(store_root, database_url=database_url)
    matches = memory_store.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=5,
            memory_types=["belief_report_artifact"],
            structured_filters={
                "source_kind": "belief_report_artifact",
                "report_as_of_date": as_of_date[:10],
                "report_model_name": report_model_name,
                "report_model_version": report_model_version,
            },
        )
    )
    if matches:
        payload = dict(matches[0].record.structured_payload)
        result_snapshot = dict(payload.get("result_snapshot", {}))
        return {
            "source": "history",
            "as_of_date": as_of_date,
            "report_model_name": report_model_name,
            "report_model_version": report_model_version,
            "report_path": dict(result_snapshot.get("named_outputs", {})).get("daily_report", {}).get("report_path"),
            "markdown": dict(result_snapshot.get("named_outputs", {})).get("daily_report", {}).get("markdown"),
            "run_result": result_snapshot,
        }

    resolved_report_root = Path(report_root or AppPaths.default().reports_dir).resolve()
    report_path = _find_existing_daily_report_path(
        report_root=resolved_report_root,
        as_of_date=as_of_date,
        report_model_name=report_model_name,
        report_model_version=report_model_version,
    )
    if report_path is None:
        return None
    return {
        "source": "legacy_markdown_history",
        "as_of_date": as_of_date,
        "report_model_name": report_model_name,
        "report_model_version": report_model_version,
        "report_path": str(report_path),
        "markdown": report_path.read_text(encoding="utf-8"),
    }


def load_or_run_historical_belief_report(
    *,
    as_of_date: str,
    input_payload: dict[str, Any],
    agent_path: str | Path | None = None,
    report_root: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
    ibkr_data_pipe=None,
) -> dict[str, Any]:
    existing = load_historical_belief_report(
        as_of_date=as_of_date,
        agent_path=agent_path,
        report_root=report_root,
        storage_root=storage_root,
        database_url=database_url,
    )
    if existing is not None:
        if existing.get("source") == "legacy_markdown_history":
            existing = None
        else:
            return existing

    resolved_report_root = Path(report_root or AppPaths.default().reports_dir).resolve()
    payload = dict(input_payload)
    payload["as_of_date"] = as_of_date
    payload["report_root"] = str(resolved_report_root)
    result = run_daily_regime_agent(
        input_payload=payload,
        agent_path=agent_path,
        storage_root=storage_root,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
        ibkr_data_pipe=ibkr_data_pipe,
    )
    daily_report = dict(result.get("named_outputs", {}).get("daily_report", {}))
    return {
        "source": "run",
        "as_of_date": as_of_date,
        "report_model_name": _resolve_report_model_identity(agent_path)[0],
        "report_model_version": _resolve_report_model_identity(agent_path)[1],
        "report_path": daily_report.get("report_path"),
        "markdown": daily_report.get("markdown"),
        "run_result": result,
    }


def _resolve_memory_namespace(agent_path: str | Path | None = None) -> str:
    resolved_agent_path = Path(agent_path or default_agent_path()).resolve()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    return agent_definition.memory_namespace or f"{agent_definition.agent_id}_memory"


def _resolve_hmm_variant_id(agent_path: str | Path | None = None) -> str:
    resolved_agent_path = Path(agent_path or default_hmm_agent_path()).resolve()
    agent_definition = YamlAgentDefinitionService().load(resolved_agent_path)
    belief_engine = str(dict(agent_definition.metadata).get("belief_engine", "hmm_gaussian_v1")).strip().lower()
    if belief_engine == "hmm_gaussian_v3":
        return "v3"
    return "v2" if belief_engine == "hmm_gaussian_v2" else "v1"


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


def reset_hmm_persisted_state(
    *,
    agent_path: str | Path | None = None,
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    app_paths: AppPaths | None = None,
) -> dict[str, Any]:
    """Delete persisted HMM memory/cache records and the saved HMM model artifact."""
    namespace = _resolve_memory_namespace(agent_path or default_hmm_agent_path())
    store_root = Path(storage_root or Path.cwd() / ".workflow_memory")
    memory_store = FilesystemMemoryStore(store_root, database_url=database_url)

    deleted_counts: dict[str, int] = {}
    deleted_counts["hmm_state_snapshot"] = memory_store.delete(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=10_000,
            memory_types=["hmm_state_snapshot"],
            structured_filters={"source_kind": "hmm_gaussian"},
        )
    )
    deleted_counts["regime_history_cache"] = memory_store.delete(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=10_000,
            memory_types=["regime_history_cache"],
            structured_filters={"source_kind": "ibkr_regime_history_cache"},
        )
    )
    deleted_counts["live_observation_snapshot"] = memory_store.delete(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=10_000,
            memory_types=["live_observation_snapshot"],
            structured_filters={"source_kind": "live_ibkr"},
        )
    )

    resolved_paths = app_paths or AppPaths.default()
    hmm_config = load_hmm_config(
        app_paths=resolved_paths,
        variant_id=_resolve_hmm_variant_id(agent_path or default_hmm_agent_path()),
    )
    model_path = resolved_paths.models_dir / "hmm" / hmm_config.model_artifact_name
    artifact_deleted = False
    if model_path.exists():
        model_path.unlink()
        artifact_deleted = True

    return {
        "namespace": namespace,
        "deleted_memory_records": deleted_counts,
        "deleted_model_artifact": artifact_deleted,
        "model_artifact_path": str(model_path),
    }


def snapshot_hmm_baseline(
    *,
    snapshot_label: str | None = None,
    agent_path: str | Path | None = None,
    app_paths: AppPaths | None = None,
) -> dict[str, Any]:
    """Copy the current HMM artifact and feature config into a versioned snapshot folder."""
    resolved_paths = app_paths or AppPaths.default()
    variant_id = _resolve_hmm_variant_id(agent_path or default_hmm_agent_path())
    hmm_config = load_hmm_config(app_paths=resolved_paths, variant_id=variant_id)
    model_path = resolved_paths.models_dir / "hmm" / hmm_config.model_artifact_name
    config_file_name = {
        "v1": "hmm_v1_core.yaml",
        "v2": "hmm_v2_core_plus_sector_corr.yaml",
        "v3": "hmm_v3_core_plus_sector_geometry.yaml",
    }.get(variant_id, "hmm_v1_core.yaml")
    config_path = resolved_paths.features_dir / config_file_name

    if not model_path.exists():
        raise FileNotFoundError(f"No trained HMM artifact exists yet at: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"No HMM feature config exists at: {config_path}")

    with model_path.open("rb") as handle:
        artifact = json.loads(json.dumps(pickle.load(handle), default=str))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    normalized_label = "".join(
        character.lower() if character.isalnum() else "_"
        for character in (snapshot_label or "baseline")
    ).strip("_") or "baseline"

    snapshot_dir = resolved_paths.models_dir / "hmm" / "snapshots" / f"{timestamp}_{normalized_label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied_model_path = snapshot_dir / hmm_config.model_artifact_name
    copied_config_path = snapshot_dir / config_path.name
    manifest_path = snapshot_dir / "snapshot_manifest.json"

    shutil.copy2(model_path, copied_model_path)
    shutil.copy2(config_path, copied_config_path)

    manifest = {
        "snapshot_created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_label": normalized_label,
        "source_model_artifact_path": str(model_path),
        "source_config_path": str(config_path),
        "model_version": artifact.get("model_version"),
        "trained_as_of": artifact.get("trained_as_of"),
        "last_trained_at": artifact.get("last_trained_at"),
        "training_row_count": artifact.get("training_row_count"),
        "train_window": artifact.get("train_window"),
        "n_components": artifact.get("n_components"),
        "covariance_type": artifact.get("covariance_type"),
        "feature_list": list(artifact.get("feature_list", [])),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "snapshot_dir": str(snapshot_dir),
        "snapshot_label": normalized_label,
        "model_artifact_path": str(copied_model_path),
        "config_path": str(copied_config_path),
        "manifest_path": str(manifest_path),
        "feature_count": len(manifest["feature_list"]),
        "training_row_count": manifest.get("training_row_count"),
        "trained_as_of": manifest.get("trained_as_of"),
    }


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
    result["agent"] = {
        "agent_id": agent_definition.agent_id,
        "name": agent_definition.name,
        "role": agent_definition.role,
        "workflow_path": agent_definition.workflow_path,
        "memory_service_type": agent_definition.memory_service_type,
        "runtime_profile": agent_definition.runtime_profile,
    }
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
