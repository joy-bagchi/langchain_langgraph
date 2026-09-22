# Baseline history contract

Contract version: `1`. The contract is portable across local filesystems and object storage. JSON is authoritative for run summaries; CSV is the portable daily-observation artifact.

## Immutable layout

```text
baselines/spy_iv_skew_vix_scaled/<sigma_multiplier_id>/<methodology_version>/<as_of_session>/<run_id>/
  summary.json
  daily_skew.csv
  report.md
```

One run may contain both DTE groups, but every observation, statistic, status, and comparison is grouped by DTE. Immutable records are never overwritten. Slice 0001 writes this layout at the configured retained local path. Until Slice 0002 establishes a GCS results prefix, weekly history exists only at that local path and is not committed.

## Summary JSON

Required top-level fields:

- `schema_version`, `methodology_version`, `instrument`, and `metric_id` (`spy_iv_skew_vix_scaled`)
- `code_commit`, canonical `configuration_hash`, `run_id`, and stable `idempotency_key`
- `as_of_session`, `market_timezone`, exchange-calendar `week_id`, `generated_at` as UTC
- `snapshot_selection_policy`, `spot_source`, `iv_units`, `sigma_multiplier`, annualization sessions (`252`), strike-distance policy, and explicit interpolation/quality policy
- historical VIX source identity and timing policy; missing historical VIX is an exclusion, never a fixed-volatility substitution
- `source_objects`: source paths plus immutable generations and/or checksums when available
- `observation_artifact`: reference and checksum for `daily_skew.csv`
- `dte_groups`: distinct objects for each DTE value and convention
- `status`: `complete`, `partial`, or `insufficient_data`
- optional `revision_of`, `supersedes`, and correction reason

Each DTE group contains trading-session DTE, its convention, retained calendar DTE/capture/expiration evidence, rolling window start/end, nominal session count, usable `n`, coverage classification and exclusions, rolling statistics, weekly statistics, and prior comparable references/differences.

Methodology `v2` strike distance uses `horizon_vol_fraction = (VIX / 100) * sqrt(trading_session_DTE / 252)` and `distance_fraction = sigma_multiplier * horizon_vol_fraction`; targets are `spot * (1 - distance_fraction)` and `spot * (1 + distance_fraction)`. Evaluate the formula directly without a rounded divisor. Skew remains put-wing IV minus call-wing IV in IV percentage points.

Statistics contain `mean`, `median`, sample `standard_deviation`, `q1`, `q3`, `minimum`, and `maximum`, all in IV percentage points. Weekly statistics summarize eligible daily implied-skew observations within the exchange-calendar week through the as-of session; they are not realized price volatility. Undefined values are JSON `null`, never zero.

The rolling reference window is the trailing configured 26 US equity trading sessions ending on the as-of session. Twenty-six is provisional and configurable, matching initial available history rather than an optimized choice. Missing sessions remain missing: do not extend the window to acquire 26 valid readings. Record nominal session count, eligible `n`, and partial-history coverage.

For each DTE, compare rolling and weekly statistics with the selected previous comparable weekly record and express differences in IV percentage points. Do not attribute a rolling-window change solely to the latest week because the window both adds and drops observations.

## Daily CSV

Use UTF-8 CSV with a stable header including: `session`, `market_timezone`, capture and expiration timestamps when available, trading-session DTE, calendar DTE, spot value/source, historical VIX value/timestamp/source, `horizon_vol_fraction`, `sigma_multiplier`, `distance_fraction`, absolute strike distance, put/call target strikes, interpolated IV values, skew in IV percentage points, eligibility, exclusion reason, input representation (fitted or raw observations), source object identities, and observation provenance/checksum fields. Rows are deterministic and ordered by session then DTE.

## Identity, revisions, and publication

Logical comparison identity is the tuple of instrument, metric, sigma multiplier, methodology version, canonical configuration, window definition, and DTE convention/value. Encode the multiplier deterministically in `sigma_multiplier_id` (for example, `sigma_0p5` or `sigma_1`). Different multipliers are separate baseline series. Never present incompatible versions as one continuous series.

Canonical identical inputs, configuration, code commit, and as-of session produce a stable idempotency key. A rerun may verify/reuse the existing logical observation but must not append a duplicate weekly record. Corrected inputs create a new immutable revision with supersession metadata; earlier snapshots remain available. Comparisons select exactly one verified revision per logical week/configuration.

Methodology `v1` used the earlier shared one-day reference distance and is superseded by horizon-scaled `v2`. Preserve any existing `v1` snapshots under their original methodology path. Generate `v2` records independently and never aggregate or compare `v1` and `v2` as one continuous series.

A replaceable latest/index manifest may point only to an entirely written, checksum-verified record with an acceptable status. Failed or incomplete writes never advance it. Updating a manifest must be conditional/atomic where supported.

The durable GCS results prefix is configurable and distinct from read-only source prefixes. This contract invents no bucket and grants no IAM. Manual backfills state their as-of date, restrict observations to data available through that date, and label whether they reconstruct corrected data retrospectively or represent a genuinely recorded historical baseline.
