# MPF canonical publication bridge

MPF must call `prepare_mpf_publication_envelope(...)` after its native JSON and Markdown are finalized and before any transport call. The bridge is producer-only and never publishes, imports no publisher client, and performs the canonical publisher's real `validate_envelope()` check locally.

Required authoritative inputs are: native forecast under `mpf_native_forecast.v1`, final Markdown, UTC finalization time, hashes of the final native JSON and Markdown, `observation.as_of`, observation source and schema, scientific model family/name/version, workflow/run/agent IDs, producer repository commit, and both canonical review flags. Missing provenance blocks publication; no default, narrative inference, or forecast timestamp substitution is permitted.

The exact callable boundary is:

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
