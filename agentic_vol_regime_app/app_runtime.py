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
from agentic_vol_regime_app.backtest_feature_store import (
    STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS,
    STRICT_10Y_COVERAGE_START,
    STRICT_10Y_TRAIN_LOOKBACK_DAYS,
    build_backtest_feature_store_from_ibkr,
)
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


def default_hmm_v3_1_agent_path() -> Path:
    return AppPaths.default().agents_dir / "daily_regime_hmm_v3_1_meta_blend_orchestrator.yaml"


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
    if belief_engine == "hmm_gaussian_v3_1":
        return ("HMMBeliefAgent", "hmm_gaussian_v3_1")
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
    if belief_engine == "hmm_gaussian_v3_1":
        return "v3_1"
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
        "v3_1": "hmm_v3_1_meta_blend.yaml",
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


def run_hmm_replay_backtester(
    *,
    config_path: str | Path,
    run_mode: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    models: list[str] | None = None,
    horizons: list[int] | None = None,
    lightweight_mode: bool = False,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Run the deterministic HMM replay backtester."""
    from src.backtest.hmm_replay.replay_config import load_replay_config
    from src.backtest.hmm_replay.replay_runner import run_hmm_replay

    resolved_config_path = Path(config_path).resolve()
    config = load_replay_config(resolved_config_path)
    services = build_platform_services(
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:run_hmm_replay_backtester",
        run_type="chain",
        inputs={
            "config_path": str(resolved_config_path),
            "run_mode": str(run_mode or getattr(config, "run_mode", "testing")).strip().lower(),
            "start_date": start_date,
            "end_date": end_date,
            "as_of_date": as_of_date,
            "models": list(models or []),
            "horizons": list(horizons or []),
            "lightweight_mode": bool(lightweight_mode),
        },
        tags=["agentic_vol_regime_app", "hmm_replay_backtester"],
        metadata={"application": "agentic_vol_regime_app"},
    ) as app_span:
        result = run_hmm_replay(
            config=config,
            run_mode=run_mode,
            start_date=start_date,
            end_date=end_date,
            as_of_date=as_of_date,
            models=models,
            horizons=horizons,
            lightweight_mode=lightweight_mode,
        )
        if hasattr(app_span, "end"):
            app_span.end(outputs=_trace_safe_payload(result))
    return result


def run_policy_backtester(
    *,
    config_path: str | Path | None = None,
    feature_store_path: str | Path | None = None,
    run_mode: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    models: list[str] | None = None,
    train_lookback_days: int | None = None,
    min_train_rows: int | None = None,
    default_dte: int | None = None,
    strike_increment: float | None = None,
    leap_delta: float | None = None,
    profit_exit_pct: float | None = None,
    loss_exit_multiple: float | None = None,
    exit_on_underlying_touch: bool | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    from src.backtest.policy.policy_backtester import (
        PolicyBacktestConfig,
        load_policy_backtest_config,
        run_policy_backtest,
    )

    base_config = (
        load_policy_backtest_config(Path(config_path).resolve())
        if config_path is not None
        else PolicyBacktestConfig()
    )
    resolved_feature_store_path = str(feature_store_path or base_config.feature_store_path)
    resolved_run_mode = str(run_mode or base_config.run_mode)
    resolved_train_lookback_days = int(
        base_config.train_lookback_days if train_lookback_days is None else train_lookback_days
    )
    resolved_min_train_rows = int(base_config.min_train_rows if min_train_rows is None else min_train_rows)
    resolved_default_dte = int(base_config.default_dte if default_dte is None else default_dte)
    resolved_strike_increment = float(base_config.strike_increment if strike_increment is None else strike_increment)
    resolved_leap_delta = float(base_config.leap_delta if leap_delta is None else leap_delta)
    resolved_profit_exit_pct = float(base_config.profit_exit_pct if profit_exit_pct is None else profit_exit_pct)
    resolved_loss_exit_multiple = float(
        base_config.loss_exit_multiple if loss_exit_multiple is None else loss_exit_multiple
    )
    resolved_exit_on_touch = bool(
        base_config.exit_on_underlying_touch if exit_on_underlying_touch is None else exit_on_underlying_touch
    )
    resolved_models = list(models if models is not None else (base_config.models or []))
    resolved_start_date = start_date or base_config.start_date
    resolved_end_date = end_date or base_config.end_date

    services = build_platform_services(
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:run_policy_backtester",
        run_type="chain",
        inputs={
            "config_path": str(config_path) if config_path is not None else "",
            "feature_store_path": str(resolved_feature_store_path),
            "run_mode": str(resolved_run_mode),
            "start_date": resolved_start_date,
            "end_date": resolved_end_date,
            "models": list(resolved_models),
            "train_lookback_days": int(resolved_train_lookback_days),
            "min_train_rows": int(resolved_min_train_rows),
            "default_dte": int(resolved_default_dte),
            "strike_increment": float(resolved_strike_increment),
            "leap_delta": float(resolved_leap_delta),
            "profit_exit_pct": float(resolved_profit_exit_pct),
            "loss_exit_multiple": float(resolved_loss_exit_multiple),
            "exit_on_underlying_touch": bool(resolved_exit_on_touch),
        },
        tags=["agentic_vol_regime_app", "policy_backtester"],
        metadata={"application": "agentic_vol_regime_app"},
    ) as app_span:
        result = run_policy_backtest(
            config=PolicyBacktestConfig(
                feature_store_path=str(resolved_feature_store_path),
                output_dir=str(base_config.output_dir),
                run_mode=str(resolved_run_mode),
                start_date=str(base_config.start_date),
                end_date=str(base_config.end_date),
                models=list(base_config.models or []),
                train_lookback_days=int(resolved_train_lookback_days),
                min_train_rows=int(resolved_min_train_rows),
                default_dte=int(resolved_default_dte),
                strike_increment=float(resolved_strike_increment),
                leap_enabled=bool(base_config.leap_enabled),
                leap_contracts=int(base_config.leap_contracts),
                leap_delta=float(resolved_leap_delta),
                leap_multiplier=int(base_config.leap_multiplier),
                risk_free_rate=float(base_config.risk_free_rate),
                dividend_yield=float(base_config.dividend_yield),
                profit_exit_pct=float(resolved_profit_exit_pct),
                loss_exit_multiple=float(resolved_loss_exit_multiple),
                exit_on_underlying_touch=bool(resolved_exit_on_touch),
                safer_reference_mode=str(base_config.safer_reference_mode),
                leap_entry_premium=float(base_config.leap_entry_premium),
                leap_profit_take_multiple=float(base_config.leap_profit_take_multiple),
                leap_stop_loss_multiple=float(base_config.leap_stop_loss_multiple),
                allow_leap_reentry=bool(base_config.allow_leap_reentry),
                allow_naked_short_calls=bool(base_config.allow_naked_short_calls),
            ),
            start_date=resolved_start_date,
            end_date=resolved_end_date,
            models=resolved_models,
            run_mode=resolved_run_mode,
        )
        if hasattr(app_span, "end"):
            app_span.end(outputs=_trace_safe_payload(result))
    return result


def run_overwrite_candidate_scorer(
    *,
    underlying: str,
    spot: float,
    vix: float,
    leap_contracts: int,
    leap_delta: float,
    candidate_csv: str | Path,
    output_dir: str | Path,
    hmm_json: str | Path | None = None,
    upside_drag_penalty: float = 0.35,
    min_premium: float = 1.40,
    max_spread_pct: float = 0.25,
    allow_crash_overwrite: bool = False,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    from agentic_vol_regime_app.overwrite_candidate_scorer import (
        ScorerConfig,
        load_candidates,
        load_hmm_context,
        score_candidates,
        write_outputs,
    )

    resolved_candidate_csv = Path(candidate_csv).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_hmm_json = Path(hmm_json).resolve() if hmm_json is not None else None
    config = ScorerConfig(
        underlying=str(underlying).strip().upper(),
        spot=float(spot),
        vix=float(vix),
        leap_contracts=int(leap_contracts),
        leap_delta=float(leap_delta),
        upside_drag_penalty=float(upside_drag_penalty),
        min_premium=float(min_premium),
        max_spread_pct=float(max_spread_pct),
        allow_crash_overwrite=bool(allow_crash_overwrite),
    )

    services = build_platform_services(
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:run_overwrite_candidate_scorer",
        run_type="chain",
        inputs={
            "underlying": config.underlying,
            "spot": config.spot,
            "vix": config.vix,
            "leap_contracts": config.leap_contracts,
            "leap_delta": config.leap_delta,
            "candidate_csv": str(resolved_candidate_csv),
            "hmm_json": str(resolved_hmm_json) if resolved_hmm_json is not None else "",
            "output_dir": str(resolved_output_dir),
            "upside_drag_penalty": config.upside_drag_penalty,
            "min_premium": config.min_premium,
            "max_spread_pct": config.max_spread_pct,
            "allow_crash_overwrite": config.allow_crash_overwrite,
        },
        tags=["agentic_vol_regime_app", "overwrite_candidate_scorer"],
        metadata={"application": "agentic_vol_regime_app"},
    ) as app_span:
        candidates = load_candidates(resolved_candidate_csv)
        hmm_context = load_hmm_context(resolved_hmm_json)
        scored_candidates, scenario_table, decision_policy = score_candidates(
            candidates,
            config=config,
            hmm_context=hmm_context,
        )
        outputs = write_outputs(
            output_dir=resolved_output_dir,
            config=config,
            hmm_context=hmm_context,
            decision_policy=decision_policy,
            scored_candidates=scored_candidates,
            scenario_table=scenario_table,
        )
        accepted = scored_candidates[scored_candidates["decision"] == "ACCEPT"].copy()
        rejected = scored_candidates[scored_candidates["decision"] == "REJECT"].copy()
        result = {
            "underlying": config.underlying,
            "spot": config.spot,
            "vix": config.vix,
            "leap_contracts": config.leap_contracts,
            "leap_delta": config.leap_delta,
            "candidate_csv": str(resolved_candidate_csv),
            "hmm_json": str(resolved_hmm_json) if resolved_hmm_json is not None else "",
            "output_dir": str(resolved_output_dir),
            "recommendation_mode": decision_policy.recommendation_mode,
            "block_new_overwrites": decision_policy.block_new_overwrites,
            "min_premium_effective": decision_policy.min_premium,
            "min_distance_sigma_effective": decision_policy.min_distance_sigma,
            "scored_candidates_path": outputs["scored_candidates_csv"],
            "scenario_pnl_path": outputs["scenario_pnl_csv"],
            "report_path": outputs["report_md"],
            "accepted_count": int(len(accepted)),
            "rejected_count": int(len(rejected)),
            "top_accepted_candidates": accepted.head(10).to_dict(orient="records"),
            "top_rejected_candidates": rejected.head(10).to_dict(orient="records"),
            "scenario_table_preview": scenario_table.head(50).to_dict(orient="records"),
            "hmm_context": (
                {
                    "asof": hmm_context.asof,
                    "selected_regime": hmm_context.selected_regime,
                    "regime_probs": dict(hmm_context.regime_probs),
                }
                if hmm_context is not None
                else None
            ),
        }
        if hasattr(app_span, "end"):
            app_span.end(outputs=_trace_safe_payload(result))
    return result


def build_backtest_feature_store(
    *,
    config_path: str | Path,
    history_days: int = 1512,
    as_of_date: str | None = None,
    symbol: str = "SPY",
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 73,
    market_data_type: int = 1,
    exchange: str = "SMART",
    option_exchange: str = "SMART",
    index_exchange: str = "CBOE",
    currency: str = "USD",
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    from src.backtest.hmm_replay.replay_config import load_replay_config

    resolved_config = Path(config_path).resolve()
    replay_config = load_replay_config(resolved_config)
    required_history_days = max(int(history_days), int(replay_config.train_lookback_days))
    if replay_config.require_10y_replay:
        required_history_days = max(required_history_days, STRICT_10Y_TRAIN_LOOKBACK_DAYS)
        if int(history_days) < STRICT_10Y_TRAIN_LOOKBACK_DAYS:
            raise RuntimeError(
                "Strict 10-year replay requires at least 2520 trading days of IBKR history for the feature-store build. "
                f"Received history_days={int(history_days)}."
            )
    output_path = replay_config.feature_store_path
    services = build_platform_services(
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    with services.observability.trace_span(
        "agentic_vol_regime_app:build_backtest_feature_store",
        run_type="chain",
        inputs={
            "config_path": str(resolved_config),
            "output_path": output_path,
            "history_days": required_history_days,
            "as_of_date": as_of_date,
            "symbol": symbol,
            "host": host,
            "port": int(port),
            "client_id": int(client_id),
            "market_data_type": int(market_data_type),
        },
        tags=["agentic_vol_regime_app", "hmm_replay_backtester", "feature_store_builder"],
        metadata={"application": "agentic_vol_regime_app"},
    ) as app_span:
        built = build_backtest_feature_store_from_ibkr(
            app_paths=AppPaths.default(),
            output_path=output_path,
            symbol=symbol,
            history_days=required_history_days,
            as_of_date=as_of_date,
            host=host,
            port=int(port),
            client_id=int(client_id),
            market_data_type=int(market_data_type),
            exchange=exchange,
            option_exchange=option_exchange,
            index_exchange=index_exchange,
            currency=currency,
            minimum_required_sector_start_date=(
                STRICT_10Y_COVERAGE_START.isoformat() if replay_config.require_10y_replay else None
            ),
        )
        result = {
            "feature_store_path": built.feature_store_path,
            "rows": built.rows,
            "start_date": built.start_date,
            "end_date": built.end_date,
            "source_as_of": built.source_as_of,
            "warnings": list(built.warnings),
            "history_coverage": list(built.history_coverage),
            "required_history_summary": dict(built.required_history_summary),
            "source_quality": dict(built.source_quality),
            "coverage_report_path": built.coverage_report_path,
        }
        if replay_config.require_10y_replay:
            built_start = datetime.fromisoformat(built.start_date).date()
            if built_start > STRICT_10Y_COVERAGE_START:
                required_summary = dict(built.required_history_summary)
                min_required_rows = int(required_summary.get("min_required_rows", 0) or 0)
                truncating_required = ", ".join(
                    str(item) for item in list(required_summary.get("truncating_required_keys", []))
                ) or "unknown"
                inferred_required_start = str(required_summary.get("inferred_required_start_date", ""))
                inferred_required_end = str(required_summary.get("inferred_required_end_date", ""))
                source_quality = dict(built.source_quality)
                missing_history = ", ".join(str(item) for item in list(source_quality.get("missing_history", []))) or "none"
                missing_symbols = ", ".join(str(item) for item in list(source_quality.get("missing_symbols", []))) or "none"
                if min_required_rows < STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS:
                    raise RuntimeError(
                        "Strict 10-year replay requires sufficient history depth. "
                        f"Feature store starts at {built_start.isoformat()} and min_required_rows={min_required_rows}, "
                        f"which is below relaxed threshold={STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}. "
                        f"Required-series inferred window: {inferred_required_start} -> {inferred_required_end}. "
                        f"Truncating required series: {truncating_required}. "
                        f"IBKR missing_history: {missing_history}. "
                        f"IBKR missing_symbols: {missing_symbols}. "
                        "The run will not fall back to a shallower history window."
                    )
                result["warnings"].append(
                    "Applied relaxed strict-replay depth policy: "
                    f"feature_store_start={built_start.isoformat()}, "
                    f"min_required_rows={min_required_rows}, "
                    f"minimum_required_rows={STRICT_REPLAY_MIN_REQUIRED_TRADING_DAYS}."
                )
        if hasattr(app_span, "end"):
            app_span.end(outputs=_trace_safe_payload(result))
    return result
