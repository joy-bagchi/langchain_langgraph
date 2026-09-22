# Slice 0001 — SPY VIX-scaled-distance skew baseline

## Outcome

Implement and run the first SPY VIX-scaled equal-distance IV-skew baseline, then save its first immutable dated local history record. Before coding, inspect actual authorized read-only GCS option-surface, historical VIX, and unadjusted-close schemas, collection timing, and any separately supplied source specification. Preserve compatible numerical and quality requirements while applying this amended VIX-scaled strike heuristic, this repository's layer boundaries, and the history contract.

## Method

- For every observation date and DTE group compute `horizon_vol_fraction = (VIX / 100) * sqrt(trading_session_DTE / 252)`, `distance_fraction = sigma_multiplier * horizon_vol_fraction`, `put_target = SPY_spot * (1 - distance_fraction)`, and `call_target = SPY_spot * (1 + distance_fraction)`. Use this formula directly rather than a rounded daily-volatility divisor.
- Default `sigma_multiplier` to `0.5`, preserving the half-daily-volatility strike-selection heuristic. It is configurable; a full daily-volatility run uses `1.0` and is a separate baseline series.
- Compute put-target IV minus call-target IV in IV percentage points. This is equal VIX-scaled distance, not matched-delta skew.
- Keep 1- and 2-trading-session DTE results separate. Retain calendar DTE and actual expiration/capture timestamps when available; derive sessions with an exchange calendar.
- Scale strike distance by trading-session DTE. For SPY `760`, VIX `15.87`, and multiplier `0.5`, approximate targets are 1-DTE: distance `$3.80`, put `756.20`, call `763.80`; 2-DTE: distance `$5.37`, put `754.63`, call `765.37`.
- Use snapshot spot, or an aligned unadjusted close only for a verified closing snapshot. Reject material timing mismatches without a suitable spot.
- Load historical VIX alongside SPY and option surfaces. Align VIX to the surface capture time; daily closing VIX is valid only for a verified closing surface. Never use today's VIX for historical snapshots, silently forward-fill a missing value, or use a VIX observation later than the snapshot. If suitable historical VIX is unavailable, exclude/report the missing input rather than substituting fixed volatility.
- Select exactly one consistent daily snapshot under a documented policy. Interpolate only within supported strike brackets; never extrapolate or substitute an unsupported expiration.
- Identify whether each input is a fitted surface or raw put/call observations and apply a source-appropriate, explicit interpolation/quality policy.
- Treat roughly 26 available dates and roughly 400 points per surface as expectations to verify, not independent observations or guaranteed counts.

Pure skew extraction/interpolation belongs in `core/volatility`; descriptive estimators belong in `core/statistics`. Dataset selection and eligibility belong in `applications/skew_baseline`; GCS/local normalization, history storage, and rendering belong in adapters.

## Results and quality evidence

Produce provenance-bearing daily observations, a machine-readable summary, and a concise human report ending with “© Cloudpulse Innovations”. Record VIX value, timestamp, source identity, trading-session DTE, `horizon_vol_fraction`, `sigma_multiplier`, `distance_fraction`, absolute strike distance, and put/call target strikes in every daily result and its provenance. For each DTE group report mean, median, sample standard deviation, quartiles, minimum, maximum, usable count, coverage, and exclusion reasons. Missing statistics remain null/absent, never zero. A latest-versus-prior comparison excludes the latest observation from its reference sample.

Write `summary.json`, `daily_skew.csv`, and `report.md` under the configured retained local history path using the immutable layout and identity rules in [the history contract](../baseline-history-contract.md). Keep different sigma multipliers in distinct baseline series and never compare them as continuous history. This first dated write is required in Slice 0001 so the result is retained before weekly cloud automation exists. Clearly report that history is local-only until Slice 0002 configures and verifies durable storage.

This horizon-scaled formula is methodology `v2` and supersedes the prior shared one-day-distance `v1` specification. If any `v1` results already exist, preserve their immutable records and generate `v2` results separately; never combine the methodologies into one history.

## Acceptance

Validate pure calculations with deterministic unit tests, adapters with schema/contract fixtures, and the application with focused end-to-end local output checks. Live reads must be authorized and read-only. Report actual coverage and exclusions; do not fabricate baseline values.
