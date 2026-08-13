"""Deterministic, consumer-facing contract for Market Physics forecasts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

FORECAST_SCHEMA_VERSION = "market_physics_forecast.v1"
FORECAST_MANIFEST_SCHEMA_VERSION = "market_physics_forecast_manifest.v1"
MPF_NATIVE_SCHEMA_VERSION = "mpf_native_forecast.v1"
MPF_INTEGRITY_SCHEMA_VERSION = "mpf_finalization_integrity.v1"
MPF_BELIEF_REPRESENTATION = "market_physics.mpf_belief_state.v1"
MPF_TRANSITION_REPRESENTATION = "market_physics.mpf_transition_probabilities.v1"
MPF_POLICY_REPRESENTATION = "market_physics.mpf_policy_recommendation.v1"
MPF_FEATURE_REPRESENTATION = "market_physics.mpf_feature_record.v1"
MPF_DATA_QUALITY_REPRESENTATION = "market_physics.mpf_data_quality.v1"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize JSON reproducibly and reject non-finite numbers."""
    def reject(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Forecast payload cannot contain NaN or Infinity.")
        if isinstance(value, dict):
            for item in value.values(): reject(item)
        elif isinstance(value, (list, tuple)):
            for item in value: reject(item)
    reject(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_envelope_sha256(envelope: dict[str, Any]) -> str:
    """Hash canonical envelope bytes with only its self-referential digest omitted."""
    snapshot = json.loads(canonical_json_bytes(envelope))
    finalization = snapshot.get("forecast", {}).get("provenance", {}).get("finalization")
    if isinstance(finalization, dict):
        finalization.pop("canonical_envelope_sha256", None)
    return sha256_bytes(canonical_json_bytes(snapshot))


def _require_text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")


def _validate_mpf_section(value: Any, *, representation: str, path: str, allow_not_applicable: bool = False, requires_context: bool = False) -> None:
    if not isinstance(value, dict) or value.get("representation") != representation:
        raise ValueError(f"{path}.representation is unsupported.")
    status = value.get("status")
    if status == "present":
        if set(value) != {"representation", "status", "scientific"}:
            raise ValueError(f"{path} has contradictory or unknown fields.")
        scientific = value.get("scientific")
        if not isinstance(scientific, dict) or not scientific:
            raise ValueError(f"{path}.scientific is required when status is present.")
        return
    if status == "unavailable":
        if set(value) != {"representation", "status", "reason", "missing_evidence"}:
            raise ValueError(f"{path} has contradictory or unknown fields.")
        _require_text(value.get("reason"), f"{path}.reason")
        missing = value.get("missing_evidence")
        if not isinstance(missing, list) or not missing or not all(isinstance(item, str) and item.strip() for item in missing):
            raise ValueError(f"{path}.missing_evidence is required when status is unavailable.")
        return
    if allow_not_applicable and status == "not_applicable":
        allowed = {"representation", "status", "reason", "scientific_context"} if requires_context else {"representation", "status", "reason"}
        if set(value) != allowed:
            raise ValueError(f"{path} has contradictory or unknown fields.")
        _require_text(value.get("reason"), f"{path}.reason")
        if requires_context:
            context = value.get("scientific_context")
            if not isinstance(context, dict) or set(context) != {"transmission_stages"}:
                raise ValueError(f"{path}.scientific_context is invalid.")
            stages = context["transmission_stages"]
            if not isinstance(stages, list) or not stages or not all(isinstance(stage, str) and stage.strip() for stage in stages):
                raise ValueError(f"{path}.scientific_context.transmission_stages is invalid.")
        return
    raise ValueError(f"{path}.status is unsupported.")


def validate_mpf_canonical_semantics(artifact: dict[str, Any]) -> None:
    """Validate additive MPF tagged sections without changing legacy HMM envelopes."""
    provenance = artifact.get("provenance")
    representations = {MPF_BELIEF_REPRESENTATION, MPF_TRANSITION_REPRESENTATION, MPF_POLICY_REPRESENTATION, MPF_FEATURE_REPRESENTATION, MPF_DATA_QUALITY_REPRESENTATION}
    model = artifact.get("model") if isinstance(artifact.get("model"), dict) else {}
    mpf_marked = (
        "scientific_payload" in artifact
        or isinstance(provenance, dict) and any(key in provenance for key in ("finalization", "native_forecast_id", "native_schema_version"))
        or any(isinstance(artifact.get(key), dict) and artifact[key].get("representation") in representations for key in ("belief_state", "transition_probabilities", "policy_recommendation", "features", "data_quality"))
        or str(model.get("belief_engine", "")).lower() in {"market_physics", "mpf"}
        or str(model.get("family", "")).lower() in {"market_physics", "mpf"}
    )
    if not mpf_marked:
        return
    if not isinstance(provenance, dict) or provenance.get("native_schema_version") != MPF_NATIVE_SCHEMA_VERSION:
        raise ValueError("forecast.provenance.native_schema_version is unsupported for an MPF artifact.")
    _validate_mpf_section(artifact.get("belief_state"), representation=MPF_BELIEF_REPRESENTATION, path="forecast.belief_state")
    _validate_mpf_section(artifact.get("transition_probabilities"), representation=MPF_TRANSITION_REPRESENTATION, path="forecast.transition_probabilities", allow_not_applicable=True, requires_context=True)
    _validate_mpf_section(artifact.get("policy_recommendation"), representation=MPF_POLICY_REPRESENTATION, path="forecast.policy_recommendation", allow_not_applicable=True)
    _validate_mpf_section(artifact.get("features"), representation=MPF_FEATURE_REPRESENTATION, path="forecast.features")
    _validate_mpf_section(artifact.get("data_quality"), representation=MPF_DATA_QUALITY_REPRESENTATION, path="forecast.data_quality")
    quality = artifact["data_quality"]
    if quality["status"] == "present" and set(quality["scientific"]) != {"source", "source_schema_version"}:
        raise ValueError("forecast.data_quality.scientific is invalid.")
    finalization = provenance.get("finalization")
    if not isinstance(finalization, dict) or finalization.get("schema_version") != MPF_INTEGRITY_SCHEMA_VERSION:
        raise ValueError("forecast.provenance.finalization is unsupported.")
    for key in ("native_forecast_sha256", "report_markdown_sha256", "canonical_envelope_sha256"):
        value = finalization.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"forecast.provenance.finalization.{key} is invalid.")
    if sha256_bytes(canonical_json_bytes(artifact.get("scientific_payload"))) != finalization["native_forecast_sha256"]:
        raise ValueError("forecast.provenance.finalization.native_forecast_sha256 does not bind scientific_payload.")


def validate_mpf_envelope_integrity(envelope: dict[str, Any]) -> None:
    """Verify all MPF finalization digests against the received transport bytes."""
    artifact, report = envelope["forecast"], envelope["report_markdown"]
    provenance = artifact.get("provenance", {})
    if provenance.get("native_schema_version") != MPF_NATIVE_SCHEMA_VERSION:
        return
    finalization = provenance["finalization"]
    if sha256_bytes(str(report).encode("utf-8")) != finalization["report_markdown_sha256"]:
        raise ValueError("forecast.provenance.finalization.report_markdown_sha256 does not bind report_markdown.")
    if canonical_envelope_sha256(envelope) != finalization["canonical_envelope_sha256"]:
        raise ValueError("forecast.provenance.finalization.canonical_envelope_sha256 does not bind envelope.")


def _utc(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("UTC timestamp is required.")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include UTC timezone.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _commit_sha() -> str | None:
    build_commit = os.getenv("BUILD_COMMIT_SHA", "").strip()
    if build_commit:
        return build_commit
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_forecast_artifact(*, named_outputs: dict[str, Any], run_id: str, workflow_id: str,
                            agent_id: str, agent_metadata: dict[str, Any], observation: dict[str, Any],
                            now: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """Build the artifact only from finalized structured workflow outputs."""
    required = ("daily_report", "belief_state", "transition_probabilities", "alert_record",
                "policy_recommendation", "critic_review", "data_quality", "feature_record")
    missing = [key for key in required if not isinstance(named_outputs.get(key), dict)]
    if missing:
        raise ValueError("Cannot publish incomplete forecast; missing finalized outputs: " + ", ".join(missing))
    belief = dict(named_outputs["belief_state"])
    hmm = dict(named_outputs.get("hmm_belief", {})) or None
    critic = dict(named_outputs["critic_review"])
    alert = dict(named_outputs["alert_record"])
    review_required = bool(critic.get("requires_human_review") or alert.get("requires_human_review"))
    model_name = str((hmm or {}).get("model_name") or agent_metadata.get("belief_engine", "heuristic"))
    model_version = str((hmm or {}).get("model_version") or belief.get("model_version", "unknown"))
    market_data_as_of = _utc(observation.get("as_of") or belief.get("as_of"))
    generated_at = _utc((now or (lambda: datetime.now(timezone.utc)))())
    core = {
        "schema_version": FORECAST_SCHEMA_VERSION, "run_id": str(run_id), "workflow_id": str(workflow_id),
        "agent_id": str(agent_id), "generated_at": generated_at, "market_data_as_of": market_data_as_of,
        "symbol": str(observation.get("symbols") and next(iter(observation["symbols"]), "SPY") or "SPY"),
        "forecast_status": "PUBLISHED", "decision_eligible": not review_required,
        "review_required": review_required, "review_status": "REQUIRED" if review_required else "NOT_REQUIRED",
        "model": {"belief_engine": str(agent_metadata.get("belief_engine", "heuristic")), "name": model_name,
                  "version": model_version, "hmm_variant": (hmm or {}).get("variant_id")},
        "data_quality": dict(named_outputs["data_quality"]), "features": dict(named_outputs["feature_record"]),
        "belief_state": belief, "hmm_belief": hmm, "transition_probabilities": dict(named_outputs["transition_probabilities"]),
        "alert": alert, "policy_recommendation": dict(named_outputs["policy_recommendation"]), "critic_review": critic,
        "provenance": {"repository_commit_sha": _commit_sha(), "input_provider": observation.get("source"),
                       "source_observation_timestamp": market_data_as_of, "workflow_id": str(workflow_id),
                       "agent_id": str(agent_id), "schema_versions": {"forecast": FORECAST_SCHEMA_VERSION,
                       "observation": observation.get("schema_version"), "belief": belief.get("schema_version")}},
    }
    # Generation time records when this artifact was made, not which forecast it is.
    # Keeping it out of the identity lets a scheduler safely reconstruct the same
    # forecast identity if a publication attempt has to be retried later.
    identity_core = dict(core)
    identity_core.pop("generated_at")
    digest = sha256_bytes(canonical_json_bytes(identity_core))
    stamp = re.sub(r"[^0-9A-Za-z]+", "-", market_data_as_of).strip("-")
    identity = re.sub(r"[^0-9A-Za-z]+", "-", f"{model_name}-{model_version}").strip("-").lower()
    core["forecast_id"] = f"forecast-{stamp}-{identity}-{digest[:16]}"
    canonical_json_bytes(core)
    return core
