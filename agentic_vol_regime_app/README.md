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

This slice is decision support only. It does not place trades, train ML models,
or promote memory into trusted priors automatically.

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

## Files

- `configs/agents/daily_regime_orchestrator.yaml` defines the example agent
- `configs/workflows/daily_belief_report.md` defines the daily workflow
- `configs/thresholds/alert_thresholds.yaml` holds alert thresholds
- `configs/features/feature_set_v1.yaml` holds deterministic feature settings
- `tests/` contains end-to-end smoke coverage for the workflow
