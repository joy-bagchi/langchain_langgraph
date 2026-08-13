from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_vol_regime_app.forecast_publisher_service import validate_envelope
from agentic_vol_regime_app.mpf_canonical_bridge import (
    MPFCanonicalizationError,
    NATIVE_SCHEMA_VERSION,
    canonical_native_sha256,
    prepare_mpf_publication_envelope,
    report_markdown_sha256,
)


def inputs():
    value = {
        "native_forecast": {
            "forecast_id": "MPF-TEST-001", "schema_version": NATIVE_SCHEMA_VERSION,
            "posterior_distributions": {"up": 0.4, "down": 0.6},
            "force_field": {"net": -0.2}, "law_assessment": {"status": "stable"},
            "scenarios": [{"name": "base"}], "observables": {"breadth": "weakening"},
            "evidence": [{"source": "ibkr"}], "uncertainty": {"level": "medium"},
            "transmission_stages": ["liquidity"], "pending_confirmations": ["close"],
            "sources": [{"name": "ibkr"}],
        },
        "report_markdown": "# Finalized MPF report",
        "finalized_at": datetime(2026, 8, 11, 20, 41, 12, 870000, tzinfo=timezone.utc),
        "observation": {"as_of": "2026-08-11T20:30:00Z", "source": "ibkr", "schema_version": "observation.v1", "symbols": {"SPY": {}}},
        "scientific_model": {"family": "market_physics", "name": "force-field", "version": "parameter-42", "variant": "v1", "parameter_version": "parameter-42"},
        "run_context": {"workflow_id": "mpf_postclose", "run_id": "run-1", "agent_id": "mpf-agent", "repository_commit": "abc123", "source_data_lineage": {"snapshot_sha256": "a" * 64}},
        "alert_record": {"requires_human_review": False},
        "critic_review": {"requires_human_review": True},
        "agent_metadata": {"llm_model": "not-the-scientific-model"},
        "scientific_semantics": {
            "transition_probabilities": {"status": "not_applicable", "reason": "MPF has no HMM transitions.", "scientific_context": {"transmission_stages": ["liquidity"]}},
            "policy_recommendation": {"status": "not_applicable", "reason": "No MPF policy is issued."},
        },
    }
    value["finalization_hashes"] = {
        "native_forecast_sha256": canonical_native_sha256(value["native_forecast"]),
        "report_markdown_sha256": report_markdown_sha256(value["report_markdown"]),
    }
    return value


def test_complete_mpf_input_is_canonical_and_preserves_native_payload():
    result = prepare_mpf_publication_envelope(**inputs())
    forecast = result.envelope["forecast"]
    assert forecast["schema_version"] == "market_physics_forecast.v1"
    assert forecast["generated_at"] == "2026-08-11T20:41:12Z"
    assert forecast["market_data_as_of"] == "2026-08-11T20:30:00Z"
    assert forecast["model"]["name"] == "force-field"
    assert forecast["model"]["name"] != inputs()["agent_metadata"]["llm_model"]
    assert forecast["scientific_payload"] == inputs()["native_forecast"]
    assert forecast["belief_state"]["representation"] == "market_physics.mpf_belief_state.v1"
    assert forecast["belief_state"]["status"] == "present"
    assert forecast["belief_state"]["scientific"]["force_field"] == {"net": -0.2}
    assert forecast["belief_state"]["scientific"]["law_assessment"] == {"status": "stable"}
    assert forecast["transition_probabilities"] == {
        "representation": "market_physics.mpf_transition_probabilities.v1",
        "status": "not_applicable",
        "reason": "MPF has no HMM transitions.",
        "scientific_context": {"transmission_stages": ["liquidity"]},
    }
    assert forecast["features"]["scientific"]["pending_confirmations"] == ["close"]
    assert len(forecast["provenance"]["finalization"]["canonical_envelope_sha256"]) == 64
    assert forecast["review_required"] is True and forecast["decision_eligible"] is False
    validate_envelope(result.envelope)


@pytest.mark.parametrize("path", ["observation", "scientific_model", "run_context", "alert_record", "critic_review"])
def test_missing_authoritative_inputs_fail_locally(path):
    value = inputs(); value[path] = {}
    with pytest.raises(MPFCanonicalizationError): prepare_mpf_publication_envelope(**value)


def test_multiple_missing_inputs_are_reported_together_and_forecast_time_is_not_a_cutoff():
    value = inputs(); value["observation"] = {"source": "", "schema_version": ""}; value["scientific_model"] = {}; value["run_context"] = {}
    with pytest.raises(MPFCanonicalizationError) as error:
        prepare_mpf_publication_envelope(**value)
    assert len(error.value.failures) >= 8
    assert all("forecast_timestamp" not in item for item in error.value.failures)


def test_source_lineage_is_required_explicitly():
    value = inputs()
    value["run_context"].pop("source_data_lineage")
    with pytest.raises(MPFCanonicalizationError) as error:
        prepare_mpf_publication_envelope(**value)
    assert "run_context.source_data_lineage is required" in str(error.value)


def test_server_finalizer_assigns_time_and_hashes_without_caller_values():
    from agentic_vol_regime_app.mpf_canonical_bridge import finalize_mpf_publication_envelope

    value = inputs()
    value.pop("finalized_at")
    value.pop("finalization_hashes")
    result = finalize_mpf_publication_envelope(
        **value,
        now=lambda: datetime(2026, 8, 11, 21, 0, 1, 999999, tzinfo=timezone.utc),
    )
    assert result.envelope["forecast"]["generated_at"] == "2026-08-11T21:00:01Z"
    assert result.envelope["forecast"]["scientific_payload"] == value["native_forecast"]


def test_missing_evidence_is_unavailable_not_not_applicable():
    value = inputs()
    for field in ("posterior_distributions", "force_field", "law_assessment", "scenarios", "observables", "evidence", "uncertainty"):
        value["native_forecast"].pop(field)
    value["finalization_hashes"]["native_forecast_sha256"] = canonical_native_sha256(value["native_forecast"])
    forecast = prepare_mpf_publication_envelope(**value).envelope["forecast"]
    assert forecast["belief_state"]["status"] == "unavailable"
    assert forecast["belief_state"]["missing_evidence"]


def test_unknown_or_malformed_mpf_semantic_tags_fail_publisher_validation():
    value = inputs()
    envelope = prepare_mpf_publication_envelope(**value).envelope
    envelope["forecast"]["belief_state"]["representation"] = "unknown.v1"
    with pytest.raises(Exception, match="representation is unsupported"):
        validate_envelope(envelope)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("belief_state", {"representation": "market_physics.mpf_belief_state.v1", "status": "present", "scientific": {}, "extra": True}),
        ("transition_probabilities", {"representation": "market_physics.mpf_transition_probabilities.v1", "status": "not_applicable", "reason": "x", "scientific_context": {"transmission_stages": []}}),
        ("policy_recommendation", {"representation": "market_physics.mpf_policy_recommendation.v1", "status": "unavailable", "reason": "x", "missing_evidence": [], "scientific": {"bad": True}}),
        ("features", {"representation": "market_physics.mpf_feature_record.v1", "status": "unavailable", "reason": "x", "missing_evidence": ["source"], "scientific": {"bad": True}}),
        ("data_quality", {"representation": "market_physics.mpf_data_quality.v1", "status": "present", "scientific": {"source": "ibkr"}}),
    ),
)
def test_strict_mpf_tagged_unions_reject_malformed_status_shapes(field, replacement):
    envelope = prepare_mpf_publication_envelope(**inputs()).envelope
    envelope["forecast"][field] = replacement
    with pytest.raises(Exception):
        validate_envelope(envelope)


def test_report_or_native_mutation_after_finalization_does_not_change_result():
    value = inputs(); result = prepare_mpf_publication_envelope(**value)
    value["native_forecast"]["posterior_distributions"]["up"] = 0.9; value["report_markdown"] = "changed"
    assert result.envelope["forecast"]["scientific_payload"]["posterior_distributions"]["up"] == 0.4
    assert result.envelope["report_markdown"] == "# Finalized MPF report"


@pytest.mark.parametrize(
    ("record", "review_required", "decision_eligible"),
    (("alert_record", True, False), ("critic_review", True, False)),
)
def test_review_governance_uses_the_canonical_or_rule(record, review_required, decision_eligible):
    value = inputs()
    value["alert_record"]["requires_human_review"] = False
    value["critic_review"]["requires_human_review"] = False
    value[record]["requires_human_review"] = True
    forecast = prepare_mpf_publication_envelope(**value).envelope["forecast"]
    assert forecast["review_required"] is review_required
    assert forecast["decision_eligible"] is decision_eligible


@pytest.mark.parametrize("target", ("native_forecast", "report_markdown"))
def test_finalization_hash_mismatch_fails_before_canonical_validation(target):
    value = inputs()
    if target == "native_forecast":
        value[target]["posterior_distributions"]["up"] = 0.9
    else:
        value[target] = "# Changed report"
    with pytest.raises(MPFCanonicalizationError) as error:
        prepare_mpf_publication_envelope(**value)
    assert "does not bind" in str(error.value)


def test_preflight_failure_never_reaches_canonical_builder(monkeypatch):
    value = inputs()
    value["observation"] = {}

    def must_not_run(**_kwargs):
        raise AssertionError("persistence or publication boundary was reached")

    monkeypatch.setattr("agentic_vol_regime_app.mpf_canonical_bridge.build_forecast_artifact", must_not_run)
    with pytest.raises(MPFCanonicalizationError):
        prepare_mpf_publication_envelope(**value)
