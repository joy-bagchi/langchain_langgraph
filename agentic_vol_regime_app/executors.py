"""Application-owned workflow executors."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentic_harness.agentic_os.tool_service import ToolExecutionRequest
from agentic_harness.contracts import MemoryQuery, MemoryRecord, StepExecutionResult, WorkflowGraphState, WorkflowStep

from agentic_vol_regime_app.alerts.predictive_alerts import build_alert_record
from agentic_vol_regime_app.config import AppPaths, load_yaml
from agentic_vol_regime_app.contracts import CriticReviewRecord, HMMBeliefRecord, ObservationRecord
from agentic_vol_regime_app.data.market_data_loader import load_market_snapshot
from agentic_vol_regime_app.data.quality import validate_observation
from agentic_vol_regime_app.features.build_features import compute_feature_record
from agentic_vol_regime_app.pomdp.belief_update import update_belief_state
from agentic_vol_regime_app.pomdp.hmm_belief import (
    compute_hmm_belief_record,
    hmm_to_belief_record,
    load_hmm_config,
)
from agentic_vol_regime_app.pomdp.ml_belief import update_belief_state_with_linear_regression
from agentic_vol_regime_app.pomdp.policy import recommend_policy_action
from agentic_vol_regime_app.pomdp.transition_model import estimate_transition_probabilities
from agentic_vol_regime_app.reports.daily_report import render_daily_markdown, write_daily_report
from agentic_vol_regime_app.features.sector_geometry import SECTOR_ETF_UNIVERSE


def _report_root(state: WorkflowGraphState, app_paths: AppPaths) -> Path:
    configured = state.get("input_payload", {}).get("report_root")
    if configured:
        return Path(str(configured)).resolve()
    return app_paths.reports_dir.resolve()


def _load_previous_belief(state: WorkflowGraphState) -> dict[str, float] | None:
    input_payload = dict(state.get("input_payload", {}))
    previous = input_payload.get("previous_belief")
    if isinstance(previous, dict):
        return {str(key): float(value) for key, value in previous.items()}
    return None


def _is_tool_allowed(state: WorkflowGraphState, tool_id: str) -> bool:
    return tool_id in {str(item) for item in state.get("allowed_tools", [])}


def _belief_engine(state: WorkflowGraphState) -> str:
    metadata = dict(state.get("agent_metadata", {}))
    return str(metadata.get("belief_engine", "heuristic")).strip().lower()


def _hmm_variant_id(state: WorkflowGraphState) -> str:
    engine = _belief_engine(state)
    if engine == "hmm_gaussian_v2":
        return "v2"
    return "v1"


def _uses_hmm_history_cache(state: WorkflowGraphState) -> bool:
    return _belief_engine(state).startswith("hmm_gaussian")


def _report_model_metadata(
    *,
    belief_record,
    hmm_record,
    state: WorkflowGraphState,
) -> tuple[str, str]:
    engine = _belief_engine(state)
    if engine.startswith("hmm_gaussian") and hmm_record is not None:
        return (str(hmm_record.model_name), str(hmm_record.model_version))
    if engine == "ml_linear_regression":
        return ("LinearRegimeBeliefModel", str(belief_record.model_version))
    return ("HeuristicBeliefModel", str(belief_record.model_version))


def _memory_namespace(state: WorkflowGraphState) -> str:
    configured = state.get("memory_namespace")
    if configured:
        return str(configured)
    return f"{state.get('agent_id') or state.get('workflow_id')}_memory"


def _load_reference_observation(
    input_payload: dict[str, Any],
    *,
    app_paths: AppPaths,
) -> ObservationRecord | None:
    reference_snapshot = input_payload.get("reference_market_snapshot")
    if isinstance(reference_snapshot, dict):
        return load_market_snapshot({"market_snapshot": reference_snapshot}, app_root=app_paths.root)

    reference_path = input_payload.get("reference_snapshot_path")
    if reference_path:
        return load_market_snapshot({"snapshot_path": reference_path}, app_root=app_paths.root)
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_regime_history_cache(*, state: WorkflowGraphState, services) -> dict[str, Any] | None:
    namespace = _memory_namespace(state)
    matches = services.memory.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=1,
            memory_types=["regime_history_cache"],
            structured_filters={"source_kind": "ibkr_regime_history_cache"},
        )
    )
    if not matches:
        return None
    payload = dict(matches[0].record.structured_payload)
    payload["record_id"] = matches[0].record.record_id
    payload["created_at"] = matches[0].record.created_at
    return payload


def _load_historical_regime_snapshot(
    *,
    state: WorkflowGraphState,
    services,
    requested_as_of_date: str,
    history_days: int,
) -> dict[str, Any] | None:
    namespace = _memory_namespace(state)
    matches = services.memory.recall(
        MemoryQuery(
            namespace=namespace,
            text="",
            max_results=25,
            memory_types=["historical_regime_snapshot"],
            structured_filters={
                "source_kind": "ibkr_historical_regime_snapshot",
                "requested_as_of_date": requested_as_of_date,
            },
        )
    )
    best_match: dict[str, Any] | None = None
    best_window = -1
    for match in matches:
        payload = dict(match.record.structured_payload)
        window = int(payload.get("history_days", 0) or 0)
        if window < history_days:
            continue
        if window > best_window:
            payload["record_id"] = match.record.record_id
            payload["created_at"] = match.record.created_at
            best_match = payload
            best_window = window
    return best_match


def _history_refresh_mode(
    *,
    cached_payload: dict[str, Any] | None,
    history_days: int,
    now: datetime,
) -> tuple[str, int]:
    if history_days <= 0:
        return ("disabled", 0)
    if not cached_payload:
        return ("full_fetch", history_days)

    cached_history = dict(cached_payload.get("history", {}))
    cached_window = int(cached_payload.get("history_days", 0) or 0)
    refreshed_at = _parse_timestamp(cached_payload.get("last_history_refresh_at"))
    if cached_window < history_days:
        return ("full_fetch", history_days)
    if refreshed_at is None:
        return ("full_fetch", history_days)
    if now - refreshed_at >= timedelta(hours=24):
        return ("incremental_refresh", 1)
    if not cached_history:
        return ("full_fetch", history_days)
    return ("cache_reuse", 0)


def _merge_history_window(
    *,
    cached_history: dict[str, list[float]] | None,
    live_history: dict[str, list[float]],
    history_days: int,
    cache_last_as_of: str | None,
    current_as_of: str | None,
) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {
        key: [float(item) for item in values]
        for key, values in dict(cached_history or {}).items()
        if values
    }
    if history_days <= 0:
        return merged

    same_observation_day = False
    cached_day = _parse_timestamp(cache_last_as_of)
    current_day = _parse_timestamp(current_as_of)
    if cached_day and current_day:
        same_observation_day = cached_day.date() == current_day.date()

    for key, live_values in dict(live_history).items():
        normalized_live = [float(item) for item in live_values if item is not None]
        if not normalized_live:
            continue
        cached_values = list(merged.get(key, []))
        if not cached_values:
            merged[key] = normalized_live[-history_days:]
            continue
        if same_observation_day:
            merged[key] = cached_values[-history_days:]
            continue
        new_value = normalized_live[-1]
        merged[key] = (cached_values + [new_value])[-history_days:]

    for key, values in list(merged.items()):
        merged[key] = values[-history_days:]
    return merged


def _remember_regime_history_cache(
    *,
    state: WorkflowGraphState,
    step: WorkflowStep,
    services,
    observation: ObservationRecord,
    history_days: int,
    refresh_mode: str,
) -> None:
    if not observation.history or history_days <= 0:
        return
    namespace = _memory_namespace(state)
    services.memory.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="regime_history_cache",
            content="latest IBKR regime history cache",
            source_run_id=str(state["run_id"]),
            source_step_id=step.step_id,
            metadata={
                "workflow_id": state.get("workflow_id"),
                "agent_id": state.get("agent_id"),
                "source_kind": "ibkr_regime_history_cache",
                "observation_as_of": observation.as_of,
            },
            structured_payload={
                "source_kind": "ibkr_regime_history_cache",
                "observation_as_of": observation.as_of,
                "last_history_refresh_at": observation.as_of,
                "history_days": history_days,
                "history_refresh_mode": refresh_mode,
                "history": {key: list(values) for key, values in observation.history.items()},
                "provider_metadata": dict(observation.provider_metadata),
            },
        )
    )


def _remember_historical_regime_snapshot(
    *,
    state: WorkflowGraphState,
    step: WorkflowStep,
    services,
    observation: ObservationRecord,
    history_days: int,
    requested_as_of_date: str,
) -> None:
    namespace = _memory_namespace(state)
    services.memory.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="historical_regime_snapshot",
            content=f"IBKR historical regime snapshot {requested_as_of_date}",
            source_run_id=str(state["run_id"]),
            source_step_id=step.step_id,
            metadata={
                "workflow_id": state.get("workflow_id"),
                "agent_id": state.get("agent_id"),
                "source_kind": "ibkr_historical_regime_snapshot",
                "requested_as_of_date": requested_as_of_date,
                "observation_as_of": observation.as_of,
            },
            structured_payload={
                "source_kind": "ibkr_historical_regime_snapshot",
                "requested_as_of_date": requested_as_of_date,
                "observation_as_of": observation.as_of,
                "history_days": history_days,
                "observation": observation.to_dict(),
            },
        )
    )


def _merge_observations(
    *,
    primary: ObservationRecord,
    fallback: ObservationRecord | None,
) -> ObservationRecord:
    if fallback is None:
        return primary

    primary_quality = validate_observation(primary)
    if primary_quality.get("is_complete", False):
        return primary

    merged_symbols: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in fallback.symbols.items()
    }
    for symbol, primary_payload in primary.symbols.items():
        merged_payload = dict(merged_symbols.get(symbol, {}))
        for field, value in dict(primary_payload).items():
            if value not in {None, ""}:
                merged_payload[field] = value
        merged_symbols[symbol] = merged_payload

    merged_history = dict(fallback.history)
    for key, values in primary.history.items():
        if values:
            merged_history[key] = list(values)

    merged_quality = dict(primary.quality)
    merged_warnings = list(dict(primary.quality).get("warnings", []))
    merged_warnings.append("Reference snapshot backfill applied for missing regime inputs.")
    merged_quality["warnings"] = merged_warnings
    merged_quality["reference_backfill_applied"] = True

    merged_provider_metadata = dict(fallback.provider_metadata)
    merged_provider_metadata.update(primary.provider_metadata)
    merged_provider_metadata["reference_backfill_source"] = fallback.source

    return ObservationRecord(
        schema_version=primary.schema_version,
        as_of=primary.as_of,
        source=f"{primary.source}+reference_backfill",
        symbols=merged_symbols,
        history=merged_history,
        quality=merged_quality,
        option_chain=dict(primary.option_chain or fallback.option_chain),
        provider_metadata=merged_provider_metadata,
    )


def _format_quality_issues(quality: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing_symbols = [str(item) for item in quality.get("missing_symbols", []) if item]
    if missing_symbols:
        issues.append(f"missing symbols: {', '.join(missing_symbols)}")
    missing_history = [str(item) for item in quality.get("missing_history", []) if item]
    if missing_history:
        issues.append(f"missing history: {', '.join(missing_history)}")
    stale_fields = [str(item) for item in quality.get("stale_fields", []) if item]
    if stale_fields:
        issues.append(f"stale fields: {', '.join(stale_fields)}")
    warnings = [str(item) for item in quality.get("warnings", []) if item]
    if warnings:
        issues.append(f"warnings: {' | '.join(warnings)}")
    return issues


def _assert_hmm_observation_complete(
    *,
    observation: ObservationRecord,
    requested_history_days: int,
) -> None:
    min_history_points = max(22, min(int(requested_history_days or 0), 252))
    quality = validate_observation(observation, min_history_points=min_history_points)
    if bool(quality.get("is_complete", False)):
        return
    details = _format_quality_issues(quality)
    message = (
        "HMM agent requires complete IBKR regime inputs and will not continue with incomplete data."
    )
    if details:
        message = f"{message} " + "; ".join(details)
    raise RuntimeError(message)


def _raise_for_untrained_hmm(*, hmm_record) -> None:
    if hmm_record.is_trained:
        return
    problems: list[str] = [f"training_status={hmm_record.training_status}"]
    warnings = [str(item) for item in hmm_record.warnings if item]
    if warnings:
        problems.append("warnings=" + " | ".join(warnings))
    drivers = [str(item) for item in hmm_record.drivers if item]
    if drivers:
        problems.append("drivers=" + " | ".join(drivers))
    raise RuntimeError(
        "HMM agent failfast: advisory model is not in a usable trained state. "
        + "; ".join(problems)
    )


def _maybe_remember_live_observation(
    *,
    state: WorkflowGraphState,
    step: WorkflowStep,
    services,
    observation: ObservationRecord,
) -> None:
    namespace = _memory_namespace(state)
    services.memory.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="live_observation_snapshot",
            content=(
                f"latest live daily observation {observation.as_of} "
                f"SPY={observation.symbols.get('SPY', {}).get('last')} "
                f"VIX={observation.symbols.get('VIX', {}).get('last')}"
            ),
            source_run_id=str(state["run_id"]),
            source_step_id=step.step_id,
            metadata={
                "workflow_id": state.get("workflow_id"),
                "agent_id": state.get("agent_id"),
                "source_kind": "live_ibkr",
                "observation_as_of": observation.as_of,
            },
            structured_payload={
                "source_kind": "live_ibkr",
                "observation_as_of": observation.as_of,
                "observation": observation.to_dict(),
            },
        )
    )


def _maybe_remember_hmm_state(
    *,
    state: WorkflowGraphState,
    step: WorkflowStep,
    services,
    hmm_record: dict[str, Any],
) -> None:
    if not bool(hmm_record.get("is_trained", False)):
        return
    namespace = _memory_namespace(state)
    top_state = str(hmm_record.get("top_state", "UNKNOWN"))
    as_of = str(hmm_record.get("as_of", ""))
    transition_probabilities = dict(hmm_record.get("transition_probabilities", {}))
    state_probabilities = dict(hmm_record.get("state_probabilities", {}))
    services.memory.remember(
        MemoryRecord.create(
            namespace=namespace,
            memory_type="hmm_state_snapshot",
            content=(
                f"HMM state snapshot {as_of} top_state={top_state} "
                f"posterior={float(state_probabilities.get(top_state, 0.0)):.2f} "
                f"expansion_or_stress_5d={float(transition_probabilities.get('to_vol_expansion_or_high_vol_5d', 0.0)):.2f}"
            ),
            source_run_id=str(state["run_id"]),
            source_step_id=step.step_id,
            metadata={
                "workflow_id": state.get("workflow_id"),
                "agent_id": state.get("agent_id"),
                "source_kind": "hmm_gaussian",
                "observation_as_of": as_of,
                "top_state": top_state,
            },
            structured_payload={
                "source_kind": "hmm_gaussian",
                "observation_as_of": as_of,
                "top_state": top_state,
                "is_trained": bool(hmm_record.get("is_trained", False)),
                "training_status": hmm_record.get("training_status"),
                "model_version": hmm_record.get("model_version"),
                "state_probabilities": state_probabilities,
                "emission_top_state": hmm_record.get("emission_top_state"),
                "emission_state_probabilities": dict(hmm_record.get("emission_state_probabilities", {})),
                "persistence_lift": dict(hmm_record.get("persistence_lift", {})),
                "transition_probabilities": transition_probabilities,
                "current_state_expected_duration_days": hmm_record.get("current_state_expected_duration_days"),
                "warnings": list(hmm_record.get("warnings", [])),
                "interpretation_notes": list(hmm_record.get("interpretation_notes", [])),
                "state_feature_summaries": dict(hmm_record.get("state_feature_summaries", {})),
                "training_row_count": hmm_record.get("training_row_count", 0),
            },
        )
    )


def build_executor_registry(*, app_paths: AppPaths, services) -> dict[str, Any]:
    """Create the app-specific executor registry."""
    threshold_config = load_yaml(app_paths.thresholds_dir / "alert_thresholds.yaml")
    feature_config = load_yaml(app_paths.features_dir / "feature_set_v1.yaml")

    def ingest_market_data(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        input_payload = dict(state.get("input_payload", {}))
        provider = str(input_payload.get("data_provider", "")).strip().lower()
        if provider == "ibkr":
            tool_id = "ibkr_data_pipeline"
            if not _is_tool_allowed(state, tool_id):
                raise RuntimeError(
                    "Daily regime orchestrator is not allowed to use the 'ibkr_data_pipeline' tool."
                )
            ibkr_payload = dict(input_payload.get("ibkr", {}))
            requested_as_of_date = str(input_payload.get("as_of_date", "")).strip() or None
            requested_history_days = max(int(ibkr_payload.get("history_days", 252)), 0)

            if _uses_hmm_history_cache(state) and requested_as_of_date:
                hmm_config = load_hmm_config(app_paths=app_paths, variant_id=_hmm_variant_id(state))
                requested_history_days = max(requested_history_days, int(hmm_config.train_window))
                historical_snapshot_payload = _load_historical_regime_snapshot(
                    state=state,
                    services=services,
                    requested_as_of_date=requested_as_of_date,
                    history_days=requested_history_days,
                )
                if historical_snapshot_payload:
                    observation = load_market_snapshot(
                        {"market_snapshot": historical_snapshot_payload["observation"]},
                        app_root=app_paths.root,
                    )
                    observation = ObservationRecord(
                        schema_version=observation.schema_version,
                        as_of=observation.as_of,
                        source=observation.source,
                        symbols=dict(observation.symbols),
                        history={key: list(values) for key, values in observation.history.items()},
                        quality=dict(observation.quality),
                        option_chain=dict(observation.option_chain),
                        provider_metadata={
                            **dict(observation.provider_metadata),
                            "history_cache_mode": "historical_memory_hit",
                            "history_cache_hit": True,
                            "history_requested_from_ibkr": 0,
                            "history_window_days": requested_history_days,
                            "requested_as_of_date": requested_as_of_date,
                        },
                    )
                    _assert_hmm_observation_complete(
                        observation=observation,
                        requested_history_days=requested_history_days,
                    )
                    return StepExecutionResult(
                        output=observation.to_dict(),
                        metadata={
                            "data_provider": "ibkr",
                            "source": observation.source,
                            "historical_as_of_date": requested_as_of_date,
                            "history_cache_mode": "historical_memory_hit",
                        },
                    )

            cached_history_payload = None
            refresh_mode = "disabled_for_engine"
            tool_history_days = requested_history_days
            if _uses_hmm_history_cache(state) and not requested_as_of_date:
                hmm_config = load_hmm_config(app_paths=app_paths, variant_id=_hmm_variant_id(state))
                requested_history_days = max(requested_history_days, int(hmm_config.train_window))
                cached_history_payload = _load_regime_history_cache(state=state, services=services)
                refresh_mode, tool_history_days = _history_refresh_mode(
                    cached_payload=cached_history_payload,
                    history_days=requested_history_days,
                    now=datetime.now(timezone.utc),
                )
            tool_arguments = {
                "operation": "fetch_vol_regime_snapshot",
                "symbol": input_payload.get("symbol", ibkr_payload.get("symbol", "SPY")),
                **ibkr_payload,
                "history_days": tool_history_days,
            }
            if requested_as_of_date:
                tool_arguments["as_of_date"] = requested_as_of_date
            if _uses_hmm_history_cache(state):
                tool_arguments["regime_symbols"] = (
                    [input_payload.get("symbol", ibkr_payload.get("symbol", "SPY")), "VIX", "VVIX", "VIX9D", "VIX3M", *SECTOR_ETF_UNIVERSE]
                    if _hmm_variant_id(state) == "v2"
                    else ibkr_payload.get("regime_symbols", [input_payload.get("symbol", ibkr_payload.get("symbol", "SPY")), "VIX", "VVIX", "VIX9D", "VIX3M", "VIX6M", "VIX9M"])
                )
            tool_response = services.tools.execute(
                ToolExecutionRequest(
                    tool_id=tool_id,
                    arguments=tool_arguments,
                    metadata={
                        "run_id": str(state.get("run_id", "")),
                        "workflow_id": str(state.get("workflow_id", "")),
                        "step_id": step.step_id,
                    },
                )
            )
            if tool_response.status != "succeeded":
                reason = str(tool_response.metadata.get("reason", "unknown tool failure"))
                raise RuntimeError(f"ibkr_data_pipeline failed: {reason}")
            if not isinstance(tool_response.output, dict):
                raise RuntimeError("ibkr_data_pipeline returned a non-dictionary snapshot payload.")
            if not dict(tool_response.output).get("symbols"):
                raise RuntimeError(
                    "ibkr_data_pipeline returned an empty snapshot without any symbols."
                )
            live_observation = load_market_snapshot(
                {"market_snapshot": tool_response.output},
                app_root=app_paths.root,
            )
            if cached_history_payload:
                merged_history = _merge_history_window(
                    cached_history=dict(cached_history_payload.get("history", {})),
                    live_history=live_observation.history,
                    history_days=requested_history_days,
                    cache_last_as_of=str(cached_history_payload.get("observation_as_of", "")),
                    current_as_of=live_observation.as_of,
                )
                live_observation = ObservationRecord(
                    schema_version=live_observation.schema_version,
                    as_of=live_observation.as_of,
                    source=live_observation.source,
                    symbols=dict(live_observation.symbols),
                    history=merged_history,
                    quality=dict(live_observation.quality),
                    option_chain=dict(live_observation.option_chain),
                    provider_metadata={
                        **dict(live_observation.provider_metadata),
                        "history_cache_mode": refresh_mode,
                        "history_cache_hit": True,
                        "history_requested_from_ibkr": tool_history_days,
                        "history_window_days": requested_history_days,
                    },
                )
            else:
                live_observation = ObservationRecord(
                    schema_version=live_observation.schema_version,
                    as_of=live_observation.as_of,
                    source=live_observation.source,
                    symbols=dict(live_observation.symbols),
                    history={key: list(values) for key, values in live_observation.history.items()},
                    quality=dict(live_observation.quality),
                    option_chain=dict(live_observation.option_chain),
                    provider_metadata={
                        **dict(live_observation.provider_metadata),
                        "history_cache_mode": refresh_mode,
                        "history_cache_hit": False,
                        "history_requested_from_ibkr": tool_history_days,
                        "history_window_days": requested_history_days,
                    },
                )
            if _uses_hmm_history_cache(state):
                observation = live_observation
                _assert_hmm_observation_complete(
                    observation=observation,
                    requested_history_days=requested_history_days,
                )
            else:
                observation = _merge_observations(
                    primary=live_observation,
                    fallback=_load_reference_observation(input_payload, app_paths=app_paths),
                )
            if _uses_hmm_history_cache(state) and requested_as_of_date:
                _remember_historical_regime_snapshot(
                    state=state,
                    step=step,
                    services=services,
                    observation=observation,
                    history_days=requested_history_days,
                    requested_as_of_date=requested_as_of_date,
                )
            elif _uses_hmm_history_cache(state) and refresh_mode in {"full_fetch", "incremental_refresh"}:
                _remember_regime_history_cache(
                    state=state,
                    step=step,
                    services=services,
                    observation=observation,
                    history_days=requested_history_days,
                    refresh_mode=refresh_mode,
                )
            if not requested_as_of_date:
                _maybe_remember_live_observation(
                    state=state,
                    step=step,
                    services=services,
                    observation=observation,
                )
            return StepExecutionResult(
                output=observation.to_dict(),
                metadata={
                    "data_provider": "ibkr",
                    "tool_id": tool_id,
                    "source": observation.source,
                    "historical_as_of_date": requested_as_of_date,
                },
            )

        observation = load_market_snapshot(input_payload, app_root=app_paths.root)
        return StepExecutionResult(output=observation.to_dict())

    def validate_data_quality(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        observation = state["named_outputs"]["observation"]
        quality = validate_observation(load_market_snapshot({"market_snapshot": observation}, app_root=app_paths.root))
        return StepExecutionResult(output=quality)

    def compute_features(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        observation = load_market_snapshot({"market_snapshot": state["named_outputs"]["observation"]}, app_root=app_paths.root)
        record = compute_feature_record(observation, feature_config=feature_config)
        return StepExecutionResult(output=record.to_dict())

    def update_belief_state_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        observation = load_market_snapshot(
            {"market_snapshot": state["named_outputs"]["observation"]},
            app_root=app_paths.root,
        )
        feature_record = compute_feature_record(
            observation,
            feature_config=feature_config,
        ).__class__(**state["named_outputs"]["feature_record"])
        engine = _belief_engine(state)
        if engine == "ml_linear_regression":
            belief_record = update_belief_state_with_linear_regression(
                feature_record,
                observation,
                previous_belief=_load_previous_belief(state),
            )
        elif engine.startswith("hmm_gaussian"):
            existing_hmm_payload = dict(state["named_outputs"].get("hmm_belief", {}))
            hmm_record = HMMBeliefRecord(**existing_hmm_payload) if existing_hmm_payload else None
            if hmm_record is None:
                hmm_record = compute_hmm_belief_record(
                    observation,
                    feature_record,
                    app_paths=app_paths,
                    variant_id=_hmm_variant_id(state),
                    force_retrain=bool(state.get("input_payload", {}).get("as_of_date")),
                )
            _raise_for_untrained_hmm(hmm_record=hmm_record)
            belief_record = hmm_to_belief_record(
                hmm_record,
                previous_belief=_load_previous_belief(state),
            )
        else:
            belief_record = update_belief_state(
                feature_record,
                previous_belief=_load_previous_belief(state),
            )
        return StepExecutionResult(output=belief_record.to_dict())

    def compute_hmm_belief_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        if not _belief_engine(state).startswith("hmm_gaussian"):
            return StepExecutionResult(output={}, metadata={"skipped": True, "reason": "non_hmm_engine"})
        observation = load_market_snapshot(
            {"market_snapshot": state["named_outputs"]["observation"]},
            app_root=app_paths.root,
        )
        feature_record = compute_feature_record(
            observation,
            feature_config=feature_config,
        ).__class__(**state["named_outputs"]["feature_record"])
        hmm_record = compute_hmm_belief_record(
            observation,
            feature_record,
            app_paths=app_paths,
            variant_id=_hmm_variant_id(state),
            force_retrain=bool(state.get("input_payload", {}).get("as_of_date")),
        )
        if _belief_engine(state).startswith("hmm_gaussian"):
            _raise_for_untrained_hmm(hmm_record=hmm_record)
        _maybe_remember_hmm_state(
            state=state,
            step=step,
            services=services,
            hmm_record=hmm_record.to_dict(),
        )
        return StepExecutionResult(output=hmm_record.to_dict())

    def estimate_transition_probabilities_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        feature_record_dict = dict(state["named_outputs"]["feature_record"])
        belief_record_dict = dict(state["named_outputs"]["belief_state"])
        feature_record = compute_feature_record(
            load_market_snapshot({"market_snapshot": state["named_outputs"]["observation"]}, app_root=app_paths.root),
            feature_config=feature_config,
        ).__class__(**feature_record_dict)
        from agentic_vol_regime_app.contracts import BeliefRecord

        belief_record = BeliefRecord(**belief_record_dict)
        transition_record = estimate_transition_probabilities(feature_record, belief_record)
        return StepExecutionResult(output=transition_record.to_dict())

    def generate_alerts(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        from agentic_vol_regime_app.contracts import BeliefRecord, FeatureRecord, TransitionProbabilityRecord

        feature_record = FeatureRecord(**dict(state["named_outputs"]["feature_record"]))
        belief_record = BeliefRecord(**dict(state["named_outputs"]["belief_state"]))
        transition_record = TransitionProbabilityRecord(**dict(state["named_outputs"]["transition_probabilities"]))
        alert_record = build_alert_record(
            feature_record,
            belief_record,
            transition_record,
            thresholds=dict(threshold_config),
        )
        return StepExecutionResult(output=alert_record.to_dict())

    def recommend_policy(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        from agentic_vol_regime_app.contracts import AlertRecord, BeliefRecord, FeatureRecord, TransitionProbabilityRecord

        feature_record = FeatureRecord(**dict(state["named_outputs"]["feature_record"]))
        belief_record = BeliefRecord(**dict(state["named_outputs"]["belief_state"]))
        transition_record = TransitionProbabilityRecord(**dict(state["named_outputs"]["transition_probabilities"]))
        alert_record = AlertRecord(**dict(state["named_outputs"]["alert_record"]))
        engine = _belief_engine(state)
        hmm_record = (
            dict(state["named_outputs"].get("hmm_belief", {}))
            if engine.startswith("hmm_gaussian")
            else {}
        )
        recommendation = recommend_policy_action(
            feature_record,
            belief_record,
            transition_record,
            alert_record,
            hmm_record=hmm_record or None,
        )
        return StepExecutionResult(output=recommendation.to_dict())

    def critic_review(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        quality = dict(state["named_outputs"]["data_quality"])
        belief_state = dict(state["named_outputs"]["belief_state"])
        alert_record = dict(state["named_outputs"]["alert_record"])
        policy = dict(state["named_outputs"]["policy_recommendation"])

        findings: list[str] = []
        verdict = "ALLOW"
        requires_human_review = False

        if not quality.get("is_complete", False):
            verdict = "ESCALATE_TO_HUMAN"
            requires_human_review = True
            findings.append("Input data quality is incomplete; results require review.")
        if float(belief_state.get("confidence", 0.0)) < 0.45:
            verdict = "ESCALATE_TO_HUMAN"
            requires_human_review = True
            findings.append("Belief confidence is low relative to the current uncertainty.")
        if alert_record.get("severity") in {"HIGH_RISK", "CRITICAL"}:
            verdict = "ESCALATE_TO_HUMAN"
            requires_human_review = True
            findings.append("Alert severity is high enough to require explicit human review.")
        if policy.get("recommended_action") in {"AGGRESSIVE_OVERWRITE", "MANUAL_REVIEW"}:
            verdict = "ESCALATE_TO_HUMAN"
            requires_human_review = True
            findings.append(
                "Policy recommendation implies a high-conviction posture and requires human review."
            )
        if not findings:
            findings.append("Deterministic checks found the daily report candidate internally consistent.")

        critic_record = CriticReviewRecord(
            schema_version="critic_review.v1",
            as_of=str(belief_state.get("as_of")),
            verdict=verdict,
            findings=findings,
            requires_human_review=requires_human_review,
            summary=" | ".join(findings),
        )
        return StepExecutionResult(output=critic_record.to_dict())

    def persist_artifacts(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        report_root = _report_root(state, app_paths)
        artifacts_dir = report_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        as_of = str(state["named_outputs"]["belief_state"]["as_of"])
        artifact_path = artifacts_dir / f"daily_regime_artifacts_{as_of[:10]}.json"
        payload = {
            "observation": state["named_outputs"]["observation"],
            "data_quality": state["named_outputs"]["data_quality"],
            "feature_record": state["named_outputs"]["feature_record"],
            "belief_state": state["named_outputs"]["belief_state"],
            "transition_probabilities": state["named_outputs"]["transition_probabilities"],
            "alert_record": state["named_outputs"]["alert_record"],
            "policy_recommendation": state["named_outputs"]["policy_recommendation"],
            "critic_review": state["named_outputs"]["critic_review"],
            "review_decision": state["named_outputs"].get("review_decision"),
        }
        hmm_payload = state["named_outputs"].get("hmm_belief")
        if hmm_payload:
            payload["hmm_belief"] = hmm_payload
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return StepExecutionResult(
            output={
                "artifact_path": str(artifact_path),
                "artifact_keys": sorted(payload.keys()),
            }
        )

    def write_memory_candidates(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        alert_record = dict(state["named_outputs"]["alert_record"])
        belief_state = dict(state["named_outputs"]["belief_state"])
        transition_record = dict(state["named_outputs"]["transition_probabilities"])
        feature_record = dict(state["named_outputs"]["feature_record"])

        should_write = alert_record.get("severity") in {"WARNING", "HIGH_RISK", "CRITICAL"}
        if not should_write and abs(float(transition_record["transition_probabilities"].get("vol_expansion_5d", 0.0))) >= 0.35:
            should_write = True

        record_ids: list[str] = []
        if should_write:
            namespace = _memory_namespace(state)
            structured_payload = {
                "as_of": belief_state.get("as_of"),
                "alert_severity": alert_record.get("severity"),
                "belief_state": belief_state.get("beliefs", {}),
                "transition_probabilities": transition_record.get("transition_probabilities", {}),
                "feature_excerpt": {
                    "vix": feature_record["features"].get("vix"),
                    "vvix_vix_z_22d": feature_record["features"].get("vvix_vix_z_22d"),
                    "term_structure_state": feature_record["features"].get("term_structure_state"),
                },
                "status": "candidate_memory",
            }
            record = services.memory.remember(
                MemoryRecord.create(
                    namespace=namespace,
                    memory_type="candidate_signal_lesson",
                    content=(
                        f"{belief_state.get('as_of')} {alert_record.get('severity')} candidate: "
                        f"expansion={transition_record['transition_probabilities'].get('vol_expansion_5d', 0.0):.2f}"
                    ),
                    source_run_id=str(state["run_id"]),
                    source_step_id=step.step_id,
                    metadata={
                        "alert_severity": alert_record.get("severity"),
                        "workflow_id": state.get("workflow_id"),
                    },
                    structured_payload=structured_payload,
                )
            )
            record_ids.append(record.record_id)
        return StepExecutionResult(
            output={
                "candidate_count": len(record_ids),
                "record_ids": record_ids,
            }
        )

    def produce_daily_report(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        from agentic_vol_regime_app.contracts import (
            AlertRecord,
            BeliefRecord,
            CriticReviewRecord,
            FeatureRecord,
            HMMBeliefRecord,
            PolicyRecommendationRecord,
            TransitionProbabilityRecord,
        )

        feature_record = FeatureRecord(**dict(state["named_outputs"]["feature_record"]))
        belief_record = BeliefRecord(**dict(state["named_outputs"]["belief_state"]))
        transition_record = TransitionProbabilityRecord(**dict(state["named_outputs"]["transition_probabilities"]))
        hmm_record_payload = dict(state["named_outputs"].get("hmm_belief", {}))
        hmm_record = HMMBeliefRecord(**hmm_record_payload) if hmm_record_payload else None
        alert_record = AlertRecord(**dict(state["named_outputs"]["alert_record"]))
        policy_record = PolicyRecommendationRecord(**dict(state["named_outputs"]["policy_recommendation"]))
        critic_record = CriticReviewRecord(**dict(state["named_outputs"]["critic_review"]))
        review_decision = state["named_outputs"].get("review_decision")
        report_model_name, report_model_version = _report_model_metadata(
            belief_record=belief_record,
            hmm_record=hmm_record,
            state=state,
        )

        engine = _belief_engine(state)
        report_hmm_record = hmm_record if engine.startswith("hmm_gaussian") else None

        markdown = render_daily_markdown(
            feature_record=feature_record,
            belief_record=belief_record,
            transition_record=transition_record,
            hmm_record=report_hmm_record,
            alert_record=alert_record,
            policy_record=policy_record,
            critic_record=critic_record,
            review_decision=review_decision,
            report_model_name=report_model_name,
            report_model_version=report_model_version,
        )
        report_path = write_daily_report(
            markdown,
            report_root=_report_root(state, app_paths),
            as_of=belief_record.as_of,
            report_model_name=report_model_name,
        )
        return StepExecutionResult(
            output={
                "report_path": str(report_path),
                "markdown": markdown,
                "top_regime": max(belief_record.beliefs, key=belief_record.beliefs.get),
                "alert_severity": alert_record.severity,
                "recommended_action": policy_record.recommended_action,
            }
        )

    return {
        "ingest_market_data": ingest_market_data,
        "validate_data_quality": validate_data_quality,
        "compute_features": compute_features,
        "compute_hmm_belief": compute_hmm_belief_executor,
        "update_belief_state": update_belief_state_executor,
        "estimate_transition_probabilities": estimate_transition_probabilities_executor,
        "generate_alerts": generate_alerts,
        "recommend_policy": recommend_policy,
        "critic_review": critic_review,
        "persist_artifacts": persist_artifacts,
        "write_memory_candidates": write_memory_candidates,
        "produce_daily_report": produce_daily_report,
    }
