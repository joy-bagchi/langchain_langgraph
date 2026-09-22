# market-physics-core

Deterministic analytics and durable weekly SPY skew-baseline history for Market Physics.

## Development

```powershell
python -m pip install -e ".[test]"
python -m pytest
market-physics-core --help
```

## Skew baseline operations

The calculation remains in core/application code; GCS publication is transactional and the scheduler runs a deterministic Cloud Run Job. Source data stays read-only. Production history is private at `gs://marketphysics-market-manifold-data/market-physics-results/skew-baseline-history/`.

```powershell
# calculate and retain locally
market-physics-core skew-baseline run --config configs/skew_baseline.example.toml
# publish an explicitly selected record
market-physics-core skew-baseline publish --as-of 2026-09-17 --retrospective
# verified latest and chronological authoritative weekly history
market-physics-core skew-baseline latest
market-physics-core skew-baseline history
# inspect all immutable corrections/revisions
market-physics-core skew-baseline history --include-revisions
# bounded retrospective backfill; never advances latest
market-physics-core skew-baseline backfill --start 2026-08-13 --end 2026-09-17
```

Each committed run has `summary.json`, `daily_skew.csv`, `report.md`, then `commit.json`; only after read-back verification are `index.json` and eligible `latest.json` updated with generation preconditions. Identical inputs are idempotent. Corrected inputs create preserved revisions. `insufficient_data` never advances latest. The initial operational minimum is 18 usable rolling observations per DTE; eligible partial coverage stays explicit.

The weekly trigger is Friday 22:30 `America/New_York`, buffered after the established post-close source workflow. Exchange-calendar logic identifies the actual final session on holiday weeks. A missing required latest session reports `waiting_for_data`; the job uses bounded Cloud Run retries, and operators can invoke the same job as a catch-up. After the 36-hour deadline, leave the prior eligible latest pointer unchanged and surface its age. Backfills are labeled retrospective and never send email.

See [the history contract](docs/baseline-history-contract.md), [Slice 0002](docs/slices/0002-weekly-baseline-history.md), and [operations](docs/skew-baseline-operations.md).
