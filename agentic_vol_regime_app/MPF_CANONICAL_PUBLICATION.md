# MPF canonical publication bridge

MPF must call `prepare_mpf_publication_envelope(...)` after its native JSON and Markdown are finalized and before any transport call. The bridge is producer-only and never publishes, imports no publisher client, and performs the canonical publisher's real `validate_envelope()` check locally.

Required authoritative inputs are: native forecast under `mpf_native_forecast.v1`, final Markdown, `observation.as_of`, observation source and schema, scientific model family/name/version, workflow/run/agent IDs, producer repository commit, and both canonical review flags. The producer finalizer assigns its truthful UTC finalization time and calculates both input hashes server-side. Missing provenance blocks publication; no default, narrative inference, or forecast timestamp substitution is permitted.

The internal producer boundary remains available for producer code that already
has a truthful finalization timestamp and hashes:

```python
validated = prepare_mpf_publication_envelope(
    native_forecast=final_native_json,
    report_markdown=final_markdown,
    finalized_at=finalization_time,
    finalization_hashes={
        "native_forecast_sha256": canonical_native_sha256(final_native_json),
        "report_markdown_sha256": report_markdown_sha256(final_markdown),
    },
    observation=admitted_observation,
    scientific_model=market_physics_model_identity,
    run_context=authoritative_run_context,
    alert_record=canonical_alert_record,
    critic_review=canonical_critic_review,
    agent_metadata=agent_metadata,
)
# Only validated.envelope may cross the publisher/MCP boundary.
```

`market_data_as_of` is `observation.as_of`: the newest admitted market data, not forecast time, finalization time, or publication time. `generated_at` is the one UTC-Z, second-precision finalization time. The scientific model identity is distinct from any LLM identity.

The bridge takes one finalization snapshot: it deep-copies the exact native artifact into `scientific_payload`, binds both inputs with SHA-256 hashes, calls `build_forecast_artifact()`, assigns one normalized UTC-Z second-precision `generated_at`, and validates the complete `{forecast, report_markdown}` envelope before it is transport-eligible. A hash mismatch or missing authoritative value returns a local `MPFCanonicalizationError` containing every detectable preflight failure.

It applies `review_required = alert_record.requires_human_review OR critic_review.requires_human_review`, and the existing canonical builder sets `decision_eligible = not review_required`. Review flags must be explicit JSON booleans; uncertainty narrative is never used as a substitute.

Only `ValidatedMPFPublicationEnvelope.envelope` may be passed to the publisher/MCP boundary. The owning MPF Agent must replace its direct native-object call with this bridge; that agent implementation is external to this repository.

For the external ChatGPT MPF Agent, use the MCP tool
`finalize_and_publish_mpf_forecast`, not this Python function and not the
retired raw `publish_market_physics_forecast` tool. It accepts native output,
Markdown, explicit observation/model/run/review provenance, and optional LLM
provenance. The producer service computes hashes and finalization time, calls
`finalize_mpf_publication_envelope`, and returns a canonical envelope only to
the adapter; only then may the thin publisher client transport it.

Native-to-canonical mapping: `scientific_payload` is the complete unchanged
`mpf_native_forecast.v1` object. MPF sections are versioned tagged values:
`market_physics.mpf_belief_state.v1`,
`market_physics.mpf_transition_probabilities.v1`,
`market_physics.mpf_policy_recommendation.v1`,
`market_physics.mpf_feature_record.v1`, and
`market_physics.mpf_data_quality.v1`. A `present` section stores authoritative
native scientific content under `scientific`; `unavailable` requires a reason
and non-empty `missing_evidence`; `not_applicable` is permitted only for MPF
transitions and policy and requires a reason. Unavailable evidence is never
encoded as not applicable. The outer `schema_version` remains
`market_physics_forecast.v1`; the native schema is recorded in payload and
provenance.

`provenance.finalization` is `mpf_finalization_integrity.v1`. It contains the
SHA-256 of canonical JSON bytes for `scientific_payload`, SHA-256 of UTF-8
Markdown bytes, and `canonical_envelope_sha256`: SHA-256 of canonical JSON
bytes for `{forecast, report_markdown}` after removing only that digest field.
The publisher recomputes all three from the received envelope before storage.

`scientific_semantics` is an authoritative required input whenever native
transition probabilities or a native policy recommendation are absent. It must
declare `unavailable` with a reason and missing-evidence list, or
`not_applicable` with a reason; transition non-applicability additionally
requires a non-empty `scientific_context.transmission_stages` list. Absence is
never interpreted as non-applicability. The agent-finalization byte limit is
`MARKET_PHYSICS_FINALIZATION_MAX_REQUEST_BYTES`, default 2 MiB, configured to
the same value in the MCP adapter and private publisher.
