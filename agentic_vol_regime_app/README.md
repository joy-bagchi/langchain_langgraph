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
- critic review with optional human gate
- artifact persistence
- candidate memory writes
- markdown daily regime report generation
- optional HMM-backed regime advisory and HMM primary agent path

This slice is decision support only. It does not place trades, train ML models,
or promote memory into trusted priors automatically.

## HMM Agent

The app now includes a third daily agent path backed by a Gaussian HMM-style
belief engine:

- `configs/agents/daily_regime_hmm_orchestrator.yaml`

The HMM engine is advisory-first:

- it infers hidden four-state volatility regimes
- estimates state persistence and expected duration
- computes 5d / 10d / 21d transition probabilities
- feeds optional duration guidance into overwrite DTE selection

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
