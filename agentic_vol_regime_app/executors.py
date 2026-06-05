"""Application-owned workflow executors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_harness.agentic_os.tool_service import ToolExecutionRequest
from agentic_harness.contracts import MemoryRecord, StepExecutionResult, WorkflowGraphState, WorkflowStep

from agentic_vol_regime_app.alerts.predictive_alerts import build_alert_record
from agentic_vol_regime_app.config import AppPaths, load_yaml
from agentic_vol_regime_app.contracts import CriticReviewRecord, ObservationRecord
from agentic_vol_regime_app.data.market_data_loader import load_market_snapshot
from agentic_vol_regime_app.data.quality import validate_observation
from agentic_vol_regime_app.features.build_features import compute_feature_record
from agentic_vol_regime_app.pomdp.belief_update import update_belief_state
from agentic_vol_regime_app.pomdp.policy import recommend_policy_action
from agentic_vol_regime_app.pomdp.transition_model import estimate_transition_probabilities
from agentic_vol_regime_app.reports.daily_report import render_daily_markdown, write_daily_report


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


def _maybe_remember_live_observation(
    *,
    state: WorkflowGraphState,
    step: WorkflowStep,
    services,
    observation: ObservationRecord,
) -> None:
    namespace = f"{state.get('agent_id') or state.get('workflow_id')}_memory"
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
            tool_response = services.tools.execute(
                ToolExecutionRequest(
                    tool_id=tool_id,
                    arguments={
                        "operation": "fetch_vol_regime_snapshot",
                        "symbol": input_payload.get("symbol", ibkr_payload.get("symbol", "SPY")),
                        **ibkr_payload,
                    },
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
            observation = _merge_observations(
                primary=live_observation,
                fallback=_load_reference_observation(input_payload, app_paths=app_paths),
            )
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
        feature_record = compute_feature_record(
            load_market_snapshot({"market_snapshot": state["named_outputs"]["observation"]}, app_root=app_paths.root),
            feature_config=feature_config,
        )
        feature_record = feature_record.__class__(**state["named_outputs"]["feature_record"])
        belief_record = update_belief_state(
            feature_record,
            previous_belief=_load_previous_belief(state),
        )
        return StepExecutionResult(output=belief_record.to_dict())

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
        recommendation = recommend_policy_action(feature_record, belief_record, transition_record, alert_record)
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
            namespace = f"{state.get('agent_id') or state.get('workflow_id')}_memory"
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
            PolicyRecommendationRecord,
            TransitionProbabilityRecord,
        )

        feature_record = FeatureRecord(**dict(state["named_outputs"]["feature_record"]))
        belief_record = BeliefRecord(**dict(state["named_outputs"]["belief_state"]))
        transition_record = TransitionProbabilityRecord(**dict(state["named_outputs"]["transition_probabilities"]))
        alert_record = AlertRecord(**dict(state["named_outputs"]["alert_record"]))
        policy_record = PolicyRecommendationRecord(**dict(state["named_outputs"]["policy_recommendation"]))
        critic_record = CriticReviewRecord(**dict(state["named_outputs"]["critic_review"]))
        review_decision = state["named_outputs"].get("review_decision")

        markdown = render_daily_markdown(
            feature_record=feature_record,
            belief_record=belief_record,
            transition_record=transition_record,
            alert_record=alert_record,
            policy_record=policy_record,
            critic_record=critic_record,
            review_decision=review_decision,
        )
        report_path = write_daily_report(
            markdown,
            report_root=_report_root(state, app_paths),
            as_of=belief_record.as_of,
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
        "update_belief_state": update_belief_state_executor,
        "estimate_transition_probabilities": estimate_transition_probabilities_executor,
        "generate_alerts": generate_alerts,
        "recommend_policy": recommend_policy,
        "critic_review": critic_review,
        "persist_artifacts": persist_artifacts,
        "write_memory_candidates": write_memory_candidates,
        "produce_daily_report": produce_daily_report,
    }
