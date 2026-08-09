# Agentic Vol Regime App

`agentic_vol_regime_app` is a separate application module built on top of
`agentic_harness`. It keeps application logic, configs, and tests isolated so
the harness can continue to evolve as the underlying Agentic OS.

## Current Slice

Milestone 1 is implemented:

- deterministic market snapshot ingestion
- data quality validation
- feature engineering
- heuristic belief-state update
- heuristic transition probabilities
- predictive alerts
- policy recommendation
- critic review that machine-labels eligibility without blocking publication
- artifact persistence
- candidate memory writes
- markdown daily regime report generation
- optional HMM-backed regime advisory and HMM primary agent path

This slice is decision support only. It does not place trades, train ML models,
or promote memory into trusted priors automatically.

## Forecast publication

Successful production `run-daily` commands publish immutable artifacts using
Application Default Credentials (no key files):

`gs://marketphysics-market-manifold-data/market-manifold/forecasts/runs/<forecast_id>/{forecast.json,report.md}`

The atomically advanced pointer is
`gs://marketphysics-market-manifold-data/market-manifold/forecasts/manifests/latest.json`.
Forecasts use `market_physics_forecast.v1`; the pointer uses
`market_physics_forecast_manifest.v1`. `forecast_status=PUBLISHED` means the
artifact was published, while `decision_eligible=false` and `review_required=true`
mean a strategy consumer must reject it without parsing prose.

Configuration precedence is workflow input, environment, then defaults:
`forecast_publish_enabled`, `forecast_gcs_bucket`, `forecast_gcs_prefix`, and
`google_cloud_project`; environment equivalents are
`MARKET_PHYSICS_FORECAST_PUBLISH_ENABLED`, `MARKET_PHYSICS_FORECAST_GCS_BUCKET`,
`MARKET_PHYSICS_FORECAST_GCS_PREFIX`, and `GOOGLE_CLOUD_PROJECT`. The CLI enables
publication by default. Direct runtime calls are local-safe unless explicitly
enabled. Historical runs never change production `latest.json` unless
`forecast_publish_historical=true` is supplied.

Retries reuse byte-identical immutable objects and use GCS generation
preconditions for `latest.json`; conflicting bytes or a newer pointer fail the
run. The runtime identity needs only object read/create/update on this forecast
prefix (for example a bucket-scoped custom role), not project-wide Owner/Editor.
MarketManifoldPhysics can later read and validate `manifests/latest.json` and its
two descriptors. No scheduler is present in this repository: schedule the shown
`run-daily` command on the existing host and alert/retry on its nonzero exit.

Redacted examples (the test fixture produces these fields deterministically):

```json
{"manifest_schema_version":"market_physics_forecast_manifest.v1","forecast_id":"forecast-2026-08-08T20-00-00Z-heuristic-v1-<hash>","forecast":{"gs_uri":"gs://.../runs/<forecast_id>/forecast.json","sha256":"<sha256>","generation":"1"},"report":{"gs_uri":"gs://.../runs/<forecast_id>/report.md","sha256":"<sha256>","generation":"2"}}
```

```json
{"schema_version":"market_physics_forecast.v1","forecast_status":"PUBLISHED","decision_eligible":false,"review_required":true,"review_status":"REQUIRED","model":{"belief_engine":"heuristic","name":"heuristic","version":"v1"}}
```

## HMM Agent

The app now includes HMM-backed daily agent paths:

- `configs/agents/daily_regime_hmm_orchestrator.yaml` for `HMMv1`
- `configs/agents/daily_regime_hmm_v2_orchestrator.yaml` for `HMMv2`

The HMM engines are advisory-first:

- it infers hidden four-state volatility regimes
- estimates state persistence and expected duration
- computes 5d / 10d / 21d transition probabilities
- feeds optional duration guidance into overwrite DTE selection

`HMMv2` extends the `HMMv1` core features with sector-correlation signals
(`avg_pairwise_corr_21d` and `first_eigenvalue_share_21d`) while keeping the
original vol-market lens intact.

The HMM dependency is optional. Install it when you want the trained HMM path:

```bash
pip install hmmlearn scikit-learn numpy
```

Without `hmmlearn`, the HMM section still renders but will warn that the model
is unavailable.

## IBKR Data Pipe

The app now includes a vendor-first IBKR data pipe for:

- SPY underlying quote data
- selected SPY option expiries
- selected strikes across calls and puts
- bid / ask / last / close / mark
- volume and open interest
- Greeks when IBKR returns them

The live path uses the optional `ib-insync` package. Install it in the
environment where you want to hit TWS or IB Gateway:

```bash
pip install ib-insync
```

Then fetch a live snapshot:

```bash
python -m agentic_vol_regime_app.cli fetch-ibkr-snapshot --symbol SPY --port 7497 --expiry-count 2 --strike-count 8
```

To save the normalized snapshot for later workflow replay:

```bash
python -m agentic_vol_regime_app.cli fetch-ibkr-snapshot --symbol SPY --output spy_snapshot.json
```

The normalized JSON can then be used as a stable input artifact even when the
workflow itself is still deterministic.

## Run The Sample Daily Workflow

From the repo root:

```bash
python -m agentic_vol_regime_app.cli run-daily --input agentic_vol_regime_app/configs/sample_inputs/daily_snapshot_watch.json
```

To inspect the raw internal workflow state:

```bash
python -m agentic_vol_regime_app.cli run-daily --input agentic_vol_regime_app/configs/sample_inputs/daily_snapshot_watch.json --output internal
```

To resume a review-gated run:

```bash
python -m agentic_vol_regime_app.cli resume --run-id <run_id> --decision approved --notes "reviewed"
```

## Streamlit Frontend

You can run the app with a Streamlit frontend. Install the UI dependency first:

```bash
pip install streamlit
```

Then launch it from the repo root:

```bash
streamlit run agentic_vol_regime_app/streamlit_app.py
```

The frontend currently supports:

- running the deterministic daily belief workflow
- selecting Heuristic, ML, or HMM daily regime agents
- selecting HMMv1 vs HMMv2 from the same report surface
- fetching a live IBKR snapshot through the `ibkr_market_data_agent`
- resuming a review-gated daily run

The IBKR panel defaults to `127.0.0.1:4001`.

## Run The IBKR Tool Agent

This example agent uses the harness toolbox directly through the
`ibkr_data_pipeline` tool. It does not use the app-owned deterministic
executors.

With a live TWS or IB Gateway on the default port `4001`:

```bash
python -m agentic_harness run-agent --agent agentic_vol_regime_app/configs/agents/ibkr_market_data_agent.yaml --input agentic_vol_regime_app/configs/sample_inputs/ibkr_spy_snapshot.json --audience agent
```

That route exercises:

- YAML agent loading
- markdown workflow execution
- tool allowlisting
- harness toolbox dispatch
- the real `ibkr_data_pipeline` tool

## Files

- `configs/agents/daily_regime_orchestrator.yaml` defines the example agent
- `configs/workflows/daily_belief_report.md` defines the daily workflow
- `configs/thresholds/alert_thresholds.yaml` holds alert thresholds
- `configs/features/feature_set_v1.yaml` holds deterministic feature settings
- `tests/` contains end-to-end smoke coverage for the workflow
