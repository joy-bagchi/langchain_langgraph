# Slice 0002 — durable weekly baseline history

## Outcome

Implement verified durable storage and weekly execution for the Slice 0001 application, reusing existing scheduling infrastructure discovered during implementation. Configure a writable GCS results prefix distinct from read-only inputs; do not invent a bucket or modify IAM without separate authorization.

## Weekly records

For each DTE, persist two separate statistics:

1. The rolling reference baseline over the trailing configurable 26 US equity trading sessions ending on the as-of session. Missing dates remain missing and do not extend the window; report nominal sessions, eligible `n`, exclusions, and partial-history coverage.
2. The weekly realized skew summary over eligible daily implied-skew observations in that exchange-calendar week through the as-of session. It usually has five sessions and may have fewer. It is not realized price volatility.

Compare each with the selected previous compatible weekly record in IV percentage points. Explain that rolling changes reflect both entering and leaving observations. Keep DTE groups and incompatible methodology/configuration identities separate.

## Execution and readiness

Run after the final market session and successful source upload for each exchange-calendar week, including holiday-shortened weeks. Do not assume Friday is a market day or hard-code an unverified upload time. A readiness gate verifies expected source completion, object identity, snapshot timing, and data coverage before calculation and manifest publication.

Provide idempotent catch-up/retry for delayed uploads and a manual backfill path. A failed run retains diagnostics but cannot advance latest/index. Backfills use only observations available through their stated as-of dates and distinguish contemporaneously recorded history from retrospective recalculation with corrected data.

## Storage

Implement the immutable record, stable idempotency key, correction/supersession, verified pointer, and selected-revision comparison behavior in [the history contract](../baseline-history-contract.md). Persist `summary.json`, `daily_skew.csv`, and `report.md` to the configured GCS results prefix only after local/contract verification. Preserve prior records.

## Acceptance

Use focused adapter and application tests plus an authorized integration verification of write, checksum/read-back, idempotent rerun, correction revision, and non-advancement on failure. Verify the reused execution environment and scheduling path. Do not claim durable weekly operation until destination, permissions, readiness timing, and scheduled invocation are all confirmed.
