"""Canonical, non-publishing bridge for finalized MPF scientific forecasts.

The MPF caller must supply authoritative run context.  This module never calls
MCP, the publisher, or storage; it only returns a validated transport envelope.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agentic_vol_regime_app.forecast_contract import build_forecast_artifact, canonical_json_bytes
from agentic_vol_regime_app.forecast_publisher_service import ServiceValidationError, validate_envelope

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
    alert = _required_mapping(alert_record, "alert_record", failures)
    critic = _required_mapping(critic_review, "critic_review", failures)
    metadata = _required_mapping(agent_metadata, "agent_metadata", failures)
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
    named_outputs = {
        "daily_report": {"markdown": report},
        "belief_state": {"as_of": observation_value["as_of"], "model_version": model_value["version"], "beliefs": {}},
        "transition_probabilities": {},
        "alert_record": {"requires_human_review": alert_review},
        "policy_recommendation": {},
        "critic_review": {"requires_human_review": critic_review_required},
        "data_quality": {"source_schema_version": observation_value["schema_version"]},
        "feature_record": {"schema_version": "mpf_finalization.v1"},
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
        "provenance": {**artifact["provenance"], "repository_commit_sha": context["repository_commit"], "native_forecast_id": native_forecast_id, "source_data_lineage": copy.deepcopy(context.get("source_data_lineage", {})), "native_schema_version": NATIVE_SCHEMA_VERSION},
        "scientific_payload": native_snapshot,
    })
    snapshot = {"forecast": artifact, "report_markdown": report}
    snapshot_hash = hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()
    try:
        validate_envelope(snapshot)
    except ServiceValidationError as exc:
        raise MPFCanonicalizationError([f"canonical validator rejected envelope: {exc}"]) from exc
    return ValidatedMPFPublicationEnvelope(envelope=copy.deepcopy(snapshot), finalization_sha256=snapshot_hash)
