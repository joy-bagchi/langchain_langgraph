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
