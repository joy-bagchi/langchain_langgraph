# Sector History Store

The old VolRegime IBKR snapshot path refetched `history_days` on each run because it treated history as a transient list window inside `ObservationRecord.history`. That list cache could reuse recent values in memory, but it was not an authoritative dated store with per-symbol watermarks, so normal runs still called IBKR again for the same history window.

This slice adds a separate offline-first historical store for sector ETFs and `SPY`:

- `bootstrap`: full historical download, explicit and intentional
- `update`: per-symbol delta fetch using each symbol's local watermark plus bounded overlap
- `offline`: local Parquet + JSON metadata read only, with zero IBKR activity

## Dataset

Default local paths:

- `agentic_vol_regime_app/data/market_history/sector_prices_daily.parquet`
- `agentic_vol_regime_app/data/market_history/sector_prices_daily.metadata.json`

Schema version:

- `sector_prices.v1`

Canonical layout:

- `date`
- `XLK`, `XLF`, `XLE`, `XLY`, `XLP`, `XLI`, `XLB`, `XLV`, `XLU`, `XLRE`
- `SPY`

Rules:

- `date` is the trading session date
- dates are ascending and unique
- prices stay null when missing
- no forward fill
- symbols are outer-joined by date
- values must be finite and strictly positive when present

## IBKR Semantics

The new low-level dated-bar path requests daily stock bars through the existing IBKR infrastructure and preserves dates instead of rebuilding them from array position.

Preferred `whatToShow` flow for sector ETFs and `SPY`:

1. `ADJUSTED_LAST`
2. `TRADES` fallback if needed

Metadata records the actual successful `whatToShow` per symbol. If a symbol falls back to `TRADES`, the store warns and does not label that history as adjusted.

## Watermarks And Overlap

`update` mode reads the existing store first and calculates a watermark per symbol:

- `last_stored_date[symbol]`

If the symbol is already current through the target completed session, no IBKR request is made for that symbol.

Otherwise the updater requests only:

- `last_stored_date - overlap`
- through `target_end_date`

Overlap exists to pick up corrected recent bars and recent adjustments. It is bounded and does not authorize refetching years of data during normal updates.

Recent internal gaps are also considered. Within the repair horizon, the updater starts the request at the earliest recent missing date instead of silently leaving the gap behind.

## Target Completed Session

The helper uses:

- explicit `--target-end-date` when provided
- otherwise an `America/New_York` weekday and market-close-cutoff approximation

Weekends roll back to the previous weekday. Without an exchange holiday calendar dependency, this is a weekday/cutoff approximation by design.

## Commands

Bootstrap:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli bootstrap \
  --start-date 2015-01-01 \
  --target-end-date 2026-07-10
```

Delta update:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli update \
  --target-end-date 2026-07-10
```

Offline inspection:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli offline --summary
```

Validation:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli validate
```

## Verifying Delta-Only Behavior

The CLI result includes per symbol:

- previous last date
- requested start and end
- bars received
- rows inserted
- overlapping rows revised
- resulting last date

The regression test for the old bug checks that an update from `T` to `T+1` never sends a large/full historical duration again. Only a small overlap-bounded duration is requested.

## Google Cloud Storage Publication

Publishing to Google Cloud Storage is a separate step from updating the local IBKR-backed store.

Mode boundaries:

- `update`: reads the local store, contacts IBKR for a bounded delta, writes local Parquet + metadata
- `offline`: reads only the local Parquet + metadata, contacts neither IBKR nor GCS
- `publish-gcs`: reads only the local Parquet + metadata, writes to GCS, never contacts IBKR
- `verify-gcs`: reads from GCS, validates the referenced dataset, never contacts IBKR

Install the GCS dependency:

```bash
pip install -e agentic_vol_regime_app[gcs]
```

Authenticate locally with Application Default Credentials:

```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
```

Create the destination bucket manually before publishing. The publisher does not create buckets automatically.

Default GCS layout:

- `gs://<bucket>/market-manifold/datasets/<dataset_id>/sector_prices_daily.parquet`
- `gs://<bucket>/market-manifold/datasets/<dataset_id>/metadata.json`
- `gs://<bucket>/market-manifold/manifests/latest.json`

The dataset id is deterministic:

- `sector-prices-<last_market_date>-<parquet_sha256_prefix>`

Identical content publishes to the same immutable dataset path. Immutable dataset objects are never silently overwritten.

Dry run:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli publish-gcs \
  --bucket <BUCKET_NAME> \
  --prefix market-manifold \
  --dry-run
```

Publish:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli publish-gcs \
  --bucket <BUCKET_NAME> \
  --prefix market-manifold \
  --project <PROJECT_ID>
```

Verify the published dataset from GCS:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli verify-gcs \
  --bucket <BUCKET_NAME> \
  --prefix market-manifold \
  --project <PROJECT_ID>
```

Environment-variable equivalents:

- `MARKET_MANIFOLD_GCP_PROJECT`
- `MARKET_MANIFOLD_GCS_BUCKET`
- `MARKET_MANIFOLD_GCS_PREFIX`

Publication sequence:

1. Load the local Parquet with the offline loader.
2. Validate the Parquet and local metadata.
3. Calculate checksums and the deterministic dataset id.
4. Upload immutable Parquet and metadata objects.
5. Verify uploaded bytes against local SHA-256 checksums.
6. Update `manifests/latest.json` only after immutable object verification succeeds.

`latest.json` is the publication pointer that the future consumer will read first. Updating local history and publishing to GCS remain separate commands by design.

## One-Command Daily Workflow

The repository now also provides a combined manual orchestration command:

```bash
python -m agentic_vol_regime_app.data.sector_history_cli update-and-publish-gcs \
  --project marketphysics \
  --bucket <BUCKET_NAME> \
  --prefix market-manifold
```

Prerequisites:

- IB Gateway or TWS is running and reachable from the local machine
- Google Application Default Credentials are configured
- a local bootstrap store already exists
- the destination GCS bucket already exists

Environment-variable defaults:

- `MARKET_MANIFOLD_GCP_PROJECT`
- `MARKET_MANIFOLD_GCS_BUCKET`
- `MARKET_MANIFOLD_GCS_PREFIX`

Sequence:

1. Resolve the target completed market session.
2. Run the existing `update` flow against the local Parquet store.
3. Download only the missing per-symbol IBKR delta plus the configured overlap.
4. Atomically rewrite the local Parquet plus metadata.
5. Reload and validate the local dataset.
6. Enforce publication-readiness checks for the configured symbol universe.
7. Publish immutable dataset objects to GCS.
8. Update `manifests/latest.json`.
9. Verify the published dataset back from GCS.

Important behavior:

- Normal reruns do not redownload full history. Already-current symbols make zero IBKR historical requests.
- Publication is blocked when required symbols are still behind the normalized target completed session.
- If the local update succeeds but cloud publication fails, the updated local Parquet is retained. Rerunning retries only the cloud portion when the local store is already current.
- `--require-adjusted-last` enables strict adjusted-only mode. In that mode any required symbol whose stored `actual_what_to_show` is `TRADES` or `MIXED` blocks publication.
- `--dry-run-publish` performs the real IBKR update and local validation, then simulates the GCS publication without remote writes or remote verification.

This remains a manually initiated command. It does not add scheduling, cloud execution, or automatic bootstrap behavior.

## SPY, VIX, and VVIX history publication

`sync-vol-regime-history-gcs` maintains a separate local history store for `SPY`, `VIX`, and `VVIX`. It uses IBKR daily bars, writes immutable objects under `market-manifold/vol-regime-history`, and defaults to the `marketphysics-market-manifold-data` bucket. VIX and VVIX are requested as CBOE index contracts; SPY is requested as a stock contract.

```bash
python -m agentic_vol_regime_app.data.sector_history_cli sync-vol-regime-history-gcs \
  --history-years 10 \
  --dry-run
```

Omit `--dry-run` to write and verify GCS objects. The command is manually initiated; it does not schedule IBKR downloads or cloud writes.
