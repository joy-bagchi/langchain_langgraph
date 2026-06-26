# Replay Preflight Failure

- requested_start_date: `2026-03-01`
- requested_end_date: `2026-06-13`
- actual_start_date: `2026-03-01`
- actual_end_date: `2026-06-12`
- earliest_feature_store_date: `2016-03-29`
- latest_feature_store_date: `2026-06-12`

## Failures
- Feature store is missing required HMMv4 columns: regime_target
- 10-year HMMv4 replay requires feature-store coverage starting on or before 2013-01-01, but earliest available date is 2016-03-29.
- 10-year HMMv4 replay requires start_date on or before 2016-01-01, but requested start_date is 2026-03-01.
- Backtest actual_start_date moved to 2026-03-01, which violates strict 10-year replay requirements.
