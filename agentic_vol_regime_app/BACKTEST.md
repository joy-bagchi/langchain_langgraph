Build a deterministic HMM backtest / historical replay engine for the Vol Regime App.

Problem:
The current "as-of date" mode is causing model/output contamination, stale overwrite strikes, and incorrect report outputs. Do not keep patching live Daily Belief Report mode. Build a separate replay engine with isolated state.

Goal:
Evaluate whether HMM v3 predicts future volatility regime outcomes better than:
1. Heuristic model
2. HMM v1
3. HMM v2
4. HMM v3

Primary experiment:
For each historical as_of date, generate model predictions using only data available up to that date, then score outcomes over:
- T+1 trading day
- T+2 trading days
- T+3 trading days
- optionally T+5 and T+10 later

Core principle:
NO FUTURE DATA LEAKAGE.
For each as_of date:
- feature windows may use data <= as_of_date only
- model training may use data <= as_of_date only
- prediction is made at as_of_date
- outcomes are measured only after as_of_date
- no live IBKR calls inside replay
- no mutation of production model artifacts
- no use of cached live report state

Create new module path:
src/backtest/hmm_replay/

Suggested files:
- src/backtest/hmm_replay/replay_runner.py
- src/backtest/hmm_replay/replay_config.py
- src/backtest/hmm_replay/replay_dataset.py
- src/backtest/hmm_replay/replay_predictions.py
- src/backtest/hmm_replay/replay_outcomes.py
- src/backtest/hmm_replay/replay_scoring.py
- src/backtest/hmm_replay/replay_report.py
- tests/test_hmm_replay_no_lookahead.py
- tests/test_hmm_replay_state_isolation.py
- tests/test_hmm_replay_outcome_scoring.py

Do not modify the production Daily Belief Report workflow except to optionally link to replay reports.

Replay inputs:
1. Historical feature store, preferably local parquet/csv:
   data/processed/features_daily.parquet

2. Historical price/vol series required for outcomes:
   - SPY or SPX close
   - VIX close
   - VVIX close
   - realized_vol_5d
   - realized_vol_21d
   - HMM feature columns
   - sector geometry features for v2/v3 if enabled

3. Model configs:
   - hmm_v1_core.yaml
   - hmm_v2_core_plus_sector_corr.yaml
   - hmm_v3_core_plus_sector_geometry.yaml
   - heuristic model config

Do not call IBKR during replay.
If historical data is missing, fail with explicit error.

Replay config:
Create YAML config:

configs/backtest/hmm_replay.yaml

Fields:
- start_date
- end_date
- train_lookback_days: 756
- min_train_rows: 504
- horizons: [1, 2, 3]
- models:
    - heuristic
    - hmm_v1_core
    - hmm_v2_core_plus_sector_corr
    - hmm_v3_core_plus_sector_geometry
- retrain_each_date: true
- covariance_type: diag
- n_components: 4 or read from each model config
- random_state: 42
- output_dir: reports/backtests/hmm_replay/
- artifact_dir: data/backtests/hmm_replay/
- freeze_policy_outputs: true

Important:
If your production app currently uses 4 regimes, preserve that. If HMM configs use 3 states, read that from model config. Do not hardcode number of states in replay.

For each as_of_date:

1. Create isolated ReplayContext:
   {
     "run_id": uuid,
     "as_of_date": date,
     "mode": "historical_replay",
     "allow_live_data": false,
     "allow_production_artifact_write": false
   }

2. Slice training data:
   train_df = feature_df[
       date >= as_of_date - train_lookback_days
       and date <= as_of_date
   ]

3. Validate:
   - train_df row count >= min_train_rows
   - all required model features present
   - no rows after as_of_date
   - no NaN/inf in selected model features after preprocessing
   - sector features available for v2/v3 if enabled

4. Fit model fresh inside replay context OR load a snapshot trained only through as_of_date.
   For initial version, fit fresh per as_of date for correctness.

5. Generate prediction using sequence ending at as_of_date.
   Do not predict using only a single row if model normally uses a recent sequence.
   Suggested:
   inference_window = last 63 rows <= as_of_date
   proba = model.predict_proba(inference_window)[-1]

6. Persist prediction record:
   {
     "run_id": "...",
     "as_of_date": "...",
     "model_name": "hmm_v3_core_plus_sector_geometry",
     "top_state": "...",
     "state_probabilities": {...},
     "transition_matrix": [[...]],
     "expected_duration_days": {...},
     "transition_probabilities": {
       "to_higher_vol_1d": ...,
       "to_higher_vol_2d": ...,
       "to_higher_vol_3d": ...
     },
     "policy_output": {
       "overwrite_posture": "...",
       "suggested_dte": ...,
       "suggested_delta": ...,
       "suggested_strike": ...
     },
     "feature_snapshot": {...},
     "warnings": [],
     "model_diagnostics": {
       "converged": true,
       "state_usage_counts": {...},
       "state_means": {...}
     }
   }

7. Freeze policy output.
   Important: policy outputs must be calculated and stored at prediction time.
   Do not recalculate suggested strike later when scoring.
   This prevents "stuck strike" or stale mutable state bugs.

8. Compute future outcomes for each horizon h:
   outcome_date = h-th trading day after as_of_date

   Outcomes:
   - spy_return_h
   - vix_change_h
   - vvix_change_h
   - rv5_change_h
   - rv21_change_h
   - vix_fell_h
   - vix_rose_h
   - vix_spike_h
   - rv_expanded_h
   - rv_compressed_h
   - realized_regime_label_h
   - next_model_top_state_h if available from future replay predictions

Default outcome definitions:
- vix_rose_h: VIX_h > VIX_asof
- vix_fell_h: VIX_h < VIX_asof
- vix_spike_h: VIX_h / VIX_asof - 1 >= 0.10 for h <= 3
- rv_expanded_h: RV21_h > RV21_asof
- rv_compressed_h: RV21_h < RV21_asof
- equity_rallied_h: SPY_return_h > 0
- equity_sold_off_h: SPY_return_h < 0

For your current example:
If HMM v3 predicted MID_VOL_CHOP and next day SPY rose while VIX/VVIX fell, that should score as supportive of MID_VOL_CHOP / non-expansion outcome.

9. Scoring:
Score each model by horizon.

For regime classification:
- Did predicted top_state align with realized_regime_label_h?
- Did predicted direction of volatility align with observed VIX/RV direction?

For probability calibration:
- Brier score for binary events:
  - vol_expansion_event_h
  - vix_spike_h
  - rv_expanded_h
  - higher_vol_transition_h

For posture:
- Was overwrite_posture consistent with realized path?
- Example:
  - NO/LIGHT overwrite is favored if SPY rallies and VIX falls.
  - MEDIUM/AGGRESSIVE overwrite is favored if SPY chops/falls or VIX/RV remains elevated.
  - This should be scored as heuristic evaluation initially, not absolute truth.

10. Outputs:
Write:
- prediction_records.jsonl
- outcome_records.jsonl
- scored_records.jsonl
- summary_metrics.csv
- replay_report.md

Report should include:

A. Overall model comparison

| Model | Horizon | Accuracy | Brier Vol Expansion | Brier VIX Spike | Avg Lead Quality | Notes |
|---|---:|---:|---:|---:|---:|---|

B. Recent as-of date comparison

| As Of | Model | Predicted State | T+1 Outcome | T+2 Outcome | T+3 Outcome | Score |
|---|---|---|---|---|---|---|

C. Model disagreement cases
List dates where HMM v3 disagreed with HMM v1/v2/heuristic.

D. HMM v3 special section
Track whether geometry features improved:
- avoided false vol-expansion calls
- detected mid-vol chop
- improved VIX-fall / RV-compression predictions
- improved overwrite posture

E. Diagnostics
- state usage counts by model
- convergence warnings
- feature missing warnings
- degenerate model warnings

Command-line interface:
Add CLI:

python -m src.backtest.hmm_replay.replay_runner \
  --config configs/backtest/hmm_replay.yaml \
  --start-date 2026-06-01 \
  --end-date 2026-06-15 \
  --models heuristic,hmm_v1_core,hmm_v2_core_plus_sector_corr,hmm_v3_core_plus_sector_geometry \
  --horizons 1,2,3

Also support single date:

python -m src.backtest.hmm_replay.replay_runner \
  --config configs/backtest/hmm_replay.yaml \
  --as-of-date 2026-06-10 \
  --horizons 1,2,3

State isolation requirements:
- replay must not use singleton live services
- replay must not reuse mutable production model instance
- replay must not write to production latest-report path
- replay must not update live memory
- replay must not call IBKR
- replay must not mutate production HMM model artifacts
- all replay outputs go under reports/backtests/hmm_replay/ and data/backtests/hmm_replay/

Testing:
Add tests that intentionally catch the bugs we saw:
1. As-of replay cannot access rows after as_of_date.
2. Replay cannot call IBKR/live data client.
3. Model artifacts are written only under backtest artifact dir.
4. Policy output is frozen at prediction time and not recomputed during scoring.
5. Running two as-of dates does not leak state from one to the other.
6. Missing sector features cause v2/v3 fallback or warning, not wrong output.
7. Transition probabilities are computed from the replay model for that date only.
8. Report does not show stale strike from prior run.
9. Re-running same replay with same random_state produces identical outputs.

Design rule:
The replay engine must be deterministic, isolated, and boring.
Correctness is more important than speed.

Implementation order:
1. ReplayDataset that loads historical feature store only.
2. ReplayContext with hard guard allow_live_data=false.
3. PredictionRecord schema.
4. OutcomeRecord schema.
5. Single as-of date replay for HMM v3 only.
6. Extend to all models.
7. Add T+1/T+2/T+3 outcome scoring.
8. Add report generation.
9. Add state-isolation tests.
10. Add model disagreement analysis.

Do not implement automatic model promotion yet.
Do not change production HMM v1/v2/v3 policy behavior.
Do not use the Daily Belief Report workflow as the replay engine.