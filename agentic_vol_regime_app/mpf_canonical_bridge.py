"""Canonical, non-publishing bridge for finalized MPF scientific forecasts.

The MPF caller must supply authoritative run context.  This module never calls
MCP, the publisher, or storage; it only returns a validated transport envelope.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from agentic_vol_regime_app.forecast_contract import (
    MPF_BELIEF_REPRESENTATION, MPF_DATA_QUALITY_REPRESENTATION,
    MPF_FEATURE_REPRESENTATION, MPF_INTEGRITY_SCHEMA_VERSION,
    MPF_POLICY_REPRESENTATION, MPF_TRANSITION_REPRESENTATION,
    build_forecast_artifact, canonical_envelope_sha256, canonical_json_bytes,
)

NATIVE_SCHEMA_VERSION = "mpf_native_forecast.v1"


class MPFCanonicalizationError(ValueError):
    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("MPF canonicalization preflight failed: " + "; ".join(failures))


@dataclass(frozen=True)
class ValidatedMPFPublicationEnvelope:
    """A transport-ready envelope; only this type may cross the MPF publish boundary."""

    envelope: dict[str, Any]
    finalization_sha256: str


def canonical_native_sha256(native_forecast: dict[str, Any]) -> str:
    """Return the stable hash the MPF finalizer must record for its native JSON."""
    return hashlib.sha256(canonical_json_bytes(native_forecast)).hexdigest()


def report_markdown_sha256(report_markdown: str) -> str:
    """Return the stable hash the MPF finalizer must record for its Markdown bytes."""
    return hashlib.sha256(report_markdown.encode("utf-8")).hexdigest()


def _required_text(value: Any, path: str, failures: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{path} is required and must be a non-empty string.")
        return ""
    return value.strip()


def _required_mapping(value: Any, path: str, failures: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        failures.append(f"{path} is required and must be a non-empty object.")
        return {}
    return dict(value)


def _required_bool(value: Any, path: str, failures: list[str]) -> bool | None:
    if type(value) is not bool:
        failures.append(f"{path} is required and must be a JSON boolean.")
        return None
    return value


def _utc_z(finalized_at: datetime) -> str:
    if finalized_at.tzinfo is None:
        raise MPFCanonicalizationError(["finalized_at must be timezone-aware UTC."])
    return finalized_at.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def prepare_mpf_publication_envelope(
    *,
    native_forecast: dict[str, Any],
    report_markdown: str,
    finalized_at: datetime,
    observation: dict[str, Any],
    scientific_model: dict[str, Any],
    run_context: dict[str, Any],
    alert_record: dict[str, Any],
    critic_review: dict[str, Any],
    agent_metadata: dict[str, Any],
    scientific_semantics: dict[str, Any],
    finalization_hashes: dict[str, Any],
) -> ValidatedMPFPublicationEnvelope:
    """Build and validate one immutable MPF publication snapshot without transport.

    ``observation.as_of`` is the only admitted-data cutoff.  A native forecast
    timestamp is intentionally not accepted as a replacement.
    """
    failures: list[str] = []
    native = _required_mapping(native_forecast, "native_forecast", failures)
    report = _required_text(report_markdown, "report_markdown", failures)
    observation_value = _required_mapping(observation, "observation", failures)
    model_value = _required_mapping(scientific_model, "scientific_model", failures)
    context = _required_mapping(run_context, "run_context", failures)
    lineage = _required_mapping(context.get("source_data_lineage"), "run_context.source_data_lineage", failures)
    alert = _required_mapping(alert_record, "alert_record", failures)
    critic = _required_mapping(critic_review, "critic_review", failures)
    metadata = _required_mapping(agent_metadata, "agent_metadata", failures)
    semantics = _required_mapping(scientific_semantics, "scientific_semantics", failures)
    hashes = _required_mapping(finalization_hashes, "finalization_hashes", failures)

    for path, value in (
        ("observation.as_of", observation_value.get("as_of")),
        ("observation.source", observation_value.get("source")),
        ("observation.schema_version", observation_value.get("schema_version")),
        ("scientific_model.family", model_value.get("family")),
        ("scientific_model.name", model_value.get("name")),
        ("scientific_model.version", model_value.get("version")),
        ("run_context.workflow_id", context.get("workflow_id")),
        ("run_context.run_id", context.get("run_id")),
        ("run_context.agent_id", context.get("agent_id")),
        ("run_context.repository_commit", context.get("repository_commit")),
        ("finalization_hashes.native_forecast_sha256", hashes.get("native_forecast_sha256")),
        ("finalization_hashes.report_markdown_sha256", hashes.get("report_markdown_sha256")),
    ):
        _required_text(value, path, failures)
    alert_review = _required_bool(alert.get("requires_human_review"), "alert_record.requires_human_review", failures)
    critic_review_required = _required_bool(critic.get("requires_human_review"), "critic_review.requires_human_review", failures)
    if native and native.get("schema_version") != NATIVE_SCHEMA_VERSION:
        failures.append(f"native_forecast.schema_version must be {NATIVE_SCHEMA_VERSION} when supplied.")
    if native and hashes.get("native_forecast_sha256") != canonical_native_sha256(native):
        failures.append("finalization_hashes.native_forecast_sha256 does not bind the supplied native_forecast.")
    if report and hashes.get("report_markdown_sha256") != report_markdown_sha256(report):
        failures.append("finalization_hashes.report_markdown_sha256 does not bind the supplied report_markdown.")
    if failures:
        raise MPFCanonicalizationError(failures)

    finalization = _utc_z(finalized_at)
    native_snapshot = copy.deepcopy(native)
    native_forecast_id = _required_text(native_snapshot.get("forecast_id"), "native_forecast.forecast_id", failures)
    if failures:
        raise MPFCanonicalizationError(failures)

    canonical_agent_metadata = {
        **metadata,
        "belief_engine": model_value["family"],
    }
    def section(rep: str, fields: tuple[str, ...]) -> dict[str, Any]:
        scientific = {field: copy.deepcopy(native_snapshot[field]) for field in fields if field in native_snapshot}
        if scientific:
            return {"representation": rep, "status": "present", "scientific": scientific}
        return {"representation": rep, "status": "unavailable", "reason": "Authoritative MPF evidence was not supplied.", "missing_evidence": list(fields)}

    belief_state = section(MPF_BELIEF_REPRESENTATION, ("belief_state", "posterior_distributions", "force_field", "law_assessment", "scenarios", "observables", "evidence", "uncertainty"))
    transition = copy.deepcopy(native_snapshot.get("transition_probabilities"))
    transition_semantics = semantics.get("transition_probabilities")
    policy_semantics = semantics.get("policy_recommendation")
    if isinstance(transition, dict) and transition:
        transition = {"representation": MPF_TRANSITION_REPRESENTATION, "status": "present", "scientific": transition}
    else:
        if not isinstance(transition_semantics, dict):
            failures.append("scientific_semantics.transition_probabilities is required when native transition probabilities are absent.")
            transition = {}
        elif transition_semantics.get("status") == "not_applicable":
            transition = {"representation": MPF_TRANSITION_REPRESENTATION, "status": "not_applicable", "reason": transition_semantics.get("reason"), "scientific_context": copy.deepcopy(transition_semantics.get("scientific_context"))}
        elif transition_semantics.get("status") == "unavailable":
            transition = {"representation": MPF_TRANSITION_REPRESENTATION, "status": "unavailable", "reason": transition_semantics.get("reason"), "missing_evidence": copy.deepcopy(transition_semantics.get("missing_evidence"))}
        else:
            failures.append("scientific_semantics.transition_probabilities.status must be not_applicable or unavailable when native transition probabilities are absent.")
            transition = {}
    policy = copy.deepcopy(native_snapshot.get("policy_recommendation"))
    if isinstance(policy, dict) and policy:
        policy = {"representation": MPF_POLICY_REPRESENTATION, "status": "present", "scientific": policy}
    else:
        if not isinstance(policy_semantics, dict):
            failures.append("scientific_semantics.policy_recommendation is required when native policy recommendation is absent.")
            policy = {}
        elif policy_semantics.get("status") in {"not_applicable", "unavailable"}:
            policy = {"representation": MPF_POLICY_REPRESENTATION, "status": policy_semantics["status"], "reason": policy_semantics.get("reason")}
            if policy_semantics["status"] == "unavailable":
                policy["missing_evidence"] = copy.deepcopy(policy_semantics.get("missing_evidence"))
        else:
            failures.append("scientific_semantics.policy_recommendation.status must be not_applicable or unavailable when native policy recommendation is absent.")
            policy = {}
    if failures:
        raise MPFCanonicalizationError(failures)
    named_outputs = {
        "daily_report": {"markdown": report},
        "belief_state": belief_state,
        "transition_probabilities": transition,
        "alert_record": {"requires_human_review": alert_review},
        "policy_recommendation": policy,
        "critic_review": {"requires_human_review": critic_review_required},
        "data_quality": {"representation": MPF_DATA_QUALITY_REPRESENTATION, "status": "present", "scientific": {"source_schema_version": observation_value["schema_version"], "source": observation_value["source"]}},
        "feature_record": section(MPF_FEATURE_REPRESENTATION, ("observables", "evidence", "uncertainty", "pending_confirmations", "sources")),
        "hmm_belief": {"model_name": model_value["name"], "model_version": model_value["version"], "variant_id": model_value.get("variant")},
    }
    artifact = build_forecast_artifact(
        named_outputs=named_outputs,
        run_id=context["run_id"],
        workflow_id=context["workflow_id"],
        agent_id=context["agent_id"],
        agent_metadata=canonical_agent_metadata,
        observation={
            "as_of": observation_value["as_of"],
            "source": observation_value["source"],
            "schema_version": observation_value["schema_version"],
            "symbols": observation_value.get("symbols", {}),
        },
        now=lambda: datetime.fromisoformat(finalization.replace("Z", "+00:00")),
    )
    artifact.update({
        "forecast_id": native_forecast_id,
        "model": {"belief_engine": model_value["family"], "name": model_value["name"], "version": model_value["version"], **({"variant": model_value["variant"]} if model_value.get("variant") else {}), **({"configuration_id": model_value["configuration_id"]} if model_value.get("configuration_id") else {}), **({"parameter_version": model_value["parameter_version"]} if model_value.get("parameter_version") else {})},
        "provenance": {**artifact["provenance"], "repository_commit_sha": context["repository_commit"], "producer_version": context.get("producer_version"), "native_forecast_id": native_forecast_id, "source_data_lineage": copy.deepcopy(lineage), "native_schema_version": NATIVE_SCHEMA_VERSION, "finalization": {"schema_version": MPF_INTEGRITY_SCHEMA_VERSION, "native_forecast_sha256": hashes["native_forecast_sha256"], "report_markdown_sha256": hashes["report_markdown_sha256"]}},
        "scientific_payload": native_snapshot,
    })
    snapshot = {"forecast": artifact, "report_markdown": report}
    snapshot["forecast"]["provenance"]["finalization"]["canonical_envelope_sha256"] = canonical_envelope_sha256(snapshot)
    snapshot_hash = snapshot["forecast"]["provenance"]["finalization"]["canonical_envelope_sha256"]
    # Delayed import avoids a service/bridge import cycle while retaining the
    # deployed service's validator as the only final authority.
    from agentic_vol_regime_app.forecast_publisher_service import ServiceValidationError, validate_envelope
    try:
        validate_envelope(snapshot)
    except ServiceValidationError as exc:
        raise MPFCanonicalizationError([f"canonical validator rejected envelope: {exc}"]) from exc
    return ValidatedMPFPublicationEnvelope(envelope=copy.deepcopy(snapshot), finalization_sha256=snapshot_hash)


def finalize_mpf_publication_envelope(
    *,
    native_forecast: dict[str, Any],
    report_markdown: str,
    observation: dict[str, Any],
    scientific_model: dict[str, Any],
    run_context: dict[str, Any],
    alert_record: dict[str, Any],
    critic_review: dict[str, Any],
    agent_metadata: dict[str, Any],
    scientific_semantics: dict[str, Any],
    now: Callable[[], datetime] | None = None,
) -> ValidatedMPFPublicationEnvelope:
    """Server-side MPF finalization boundary; callers never supply hashes or time."""
    final_native = copy.deepcopy(native_forecast)
    final_report = report_markdown if isinstance(report_markdown, str) else report_markdown
    return prepare_mpf_publication_envelope(
        native_forecast=final_native,
        report_markdown=final_report,
        finalized_at=(now or (lambda: datetime.now(timezone.utc)))(),
        observation=observation,
        scientific_model=scientific_model,
        run_context=run_context,
        alert_record=alert_record,
        critic_review=critic_review,
        agent_metadata=agent_metadata,
        scientific_semantics=scientific_semantics,
        finalization_hashes={
            "native_forecast_sha256": canonical_native_sha256(final_native) if isinstance(final_native, dict) else None,
            "report_markdown_sha256": report_markdown_sha256(final_report) if isinstance(final_report, str) else None,
        },
    )
