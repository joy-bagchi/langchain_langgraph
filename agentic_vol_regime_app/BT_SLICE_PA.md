BACKTEST_TWEAK_SLICE_4: Add HMMv4 Path-Aware Meta Learner to Replay + Force 10-Year Backtest

Context:
We now have:
- Heuristic
- HMMv1 core vol model
- HMMv2 core vol + sector correlation
- HMMv3 core vol + sector geometry
- HMMv3.1 smooth meta-blend

Next step:
Add HMMv4 Path-Aware Meta Learner to the backtest/replay framework so we can compare its behavior against all prior models.

HMMv4 concept:
HMMv4 should not be a static point-in-time blend. It should learn from:
- current vol/geometry levels
- recent path
- slopes
- acceleration / curvature
- persistence
- vol/geometry divergence
- ensemble disagreement behavior
- prior model predictions versus realized outcomes

Goal:
Evaluate whether path-aware nonlinear learning improves:
- risk bucket prediction
- VIX spike prediction
- higher-vol transition prediction
- false alarm filtering
- missed risk reduction
- model trust behavior

IMPORTANT:
This slice must include a 10-year backtest. Do not silently reduce to 3 years. Do not default to 3 years. Do not cap the run at 3 years unless explicitly configured by the user.

Required backtest range:
- start_date: 2016-01-01
- end_date: latest available completed trading day in feature store
- minimum acceptable start_date: 2016-01-01
- if historical data does not exist back to 2016-01-01, fail loudly with an explicit error and missing data report
- do not silently shift start_date forward
- do not silently fall back to 3 years

Required train lookback:
- train_lookback_days: 2520 trading days
- min_train_rows: 1260
- HMMv4 walk_forward_train_lookback_days: 2520
- HMMv4 min_training_rows: 1260

Rationale:
We need 10 years because HMMv4 is path-aware and supervised. It requires multiple volatility regimes, vol compression periods, vol expansion periods, panic episodes, and post-panic normalization periods. A 3-year window is not enough to evaluate path-dependence or geometry curvature robustly.

Models to include:
- heuristic
- hmm_v1_core
- hmm_v2_core_plus_sector_corr
- hmm_v3_core_plus_sector_geometry
- hmm_v3_1_meta_blend
- hmm_v4_path_aware_meta

Horizons:
- 1
- 2
- 3
- 5
- 10

Primary target for HMMv4:
- realized_risk_bucket_h
  LOW_RISK vs HIGHER_VOL_RISK

Secondary targets:
- realized_state_h
- vix_spike_h
- rv_expanded_h
- higher_vol_transition_h

Do not use random train/test split.
Use walk-forward training only.

For each as_of_date:
1. Build training dataset using only rows strictly before as_of_date.
2. Use historical outcomes only if those outcomes would have been known as of as_of_date.
3. Predict for as_of_date.
4. Score against future T+1/T+2/T+3/T+5/T+10 outcomes.
5. Do not leak future realized labels into feature computation.

Feature Store Requirements:
Before running the 10-year backtest, validate that the feature store covers:
- SPY
- VIX
- VVIX
- VIX9D
- VIX3M
- sector ETFs:
  - XLK
  - XLF
  - XLE
  - XLY
  - XLP
  - XLI
  - XLB
  - XLV
  - XLU
  - XLRE

Required feature-store coverage:
- must cover from at least 2013-01-01 if using 2520-day lookback and 2016-01-01 replay start
- if using calendar days instead of trading rows, ensure enough history exists to provide 2520 valid trading rows before 2016-01-01
- if insufficient pre-2016 training history exists, either:
  A. build/refresh feature store back to 2010-01-01, or
  B. fail loudly and output exact earliest available date per symbol/feature

Do not silently proceed with fewer rows.

Add preflight validation:
Create:
- src/backtest/hmm_replay/preflight.py

Preflight must check:
1. earliest date in feature store
2. latest date in feature store
3. date coverage by required symbol/feature family
4. missing date counts
5. required rows for training lookback
6. required rows for HMMv4 supervised target construction
7. whether each model can run over requested range
8. whether HMMv4 has enough matured labels for each horizon

If validation fails:
- stop the run
- write preflight_failure_report.md
- write preflight_missing_data.csv
- do not run partial 3-year backtest

HMMv4 Path-Aware Feature Set:

Use current values:
- VIX
- VVIX
- VVIX/VIX
- VIX/VIX3M
- VIX9D/VIX
- realized_vol_5d
- realized_vol_21d
- drawdown_21d
- trend_persistence_21d
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d
- geometry_stress_score
- avg_corr_stress
- eigen_stress
- effective_rank_stress
- log_det_stress
- core_vol_risk_score
- final_risk_score from HMMv3.1

Path features:
For windows [1, 3, 5, 10, 21, 63]:

Vol features:
- vix_delta_{w}d
- vvix_delta_{w}d
- vvix_vix_ratio_delta_{w}d
- vix_vix3m_ratio_delta_{w}d
- vix9d_vix_ratio_delta_{w}d
- realized_vol_21d_delta_{w}d
- drawdown_21d_delta_{w}d
- trend_persistence_21d_delta_{w}d

Geometry features:
- geometry_stress_delta_{w}d
- avg_corr_stress_delta_{w}d
- eigen_stress_delta_{w}d
- effective_rank_stress_delta_{w}d
- log_det_stress_delta_{w}d
- avg_pairwise_corr_delta_{w}d
- first_eigenvalue_share_delta_{w}d
- effective_rank_delta_{w}d
- log_det_corr_delta_{w}d

Slope features:
For windows [3, 5, 10, 21]:
- geometry_stress_slope_{w}d
- eigen_stress_slope_{w}d
- effective_rank_stress_slope_{w}d
- log_det_stress_slope_{w}d
- vix_slope_{w}d
- vvix_slope_{w}d
- realized_vol_21d_slope_{w}d

Curvature / acceleration:
- geometry_stress_curvature_5_10
- geometry_stress_curvature_10_21
- eigen_stress_curvature_5_10
- effective_rank_stress_curvature_5_10
- log_det_stress_curvature_5_10
- vix_curvature_5_10
- vvix_curvature_5_10
- realized_vol_21d_curvature_5_10

Definition:
curvature_5_10 =
  ((x[t] - x[t-5]) / 5)
  -
  ((x[t-5] - x[t-10]) / 5)

Persistence features:
For windows [5, 10, 21, 63]:
- geometry_days_above_0_55_{w}d
- geometry_days_above_0_70_{w}d
- geometry_days_below_0_30_{w}d
- geometry_mean_{w}d
- geometry_max_{w}d
- geometry_min_{w}d
- geometry_std_{w}d
- vix_days_above_20_{w}d
- vix_days_above_25_{w}d
- vix_days_below_18_{w}d
- vvix_vix_days_rising_{w}d

Divergence features:
- vol_geometry_gap = core_vol_risk_score - geometry_stress_score
- vol_geometry_gap_abs
- vol_geometry_gap_delta_5d
- vol_geometry_gap_delta_10d
- vol_geometry_confirming
- vol_geometry_diverging

Definitions:
vol_geometry_confirming = 1 if:
  core_vol_risk_score >= 0.55
  and geometry_stress_score >= 0.55

vol_geometry_diverging = 1 if:
  core_vol_risk_score >= 0.60 and geometry_stress_score <= 0.35
  OR
  geometry_stress_score >= 0.60 and core_vol_risk_score <= 0.35

Ensemble output features:
For each model:
- top state severity
- state probabilities where available
- predicted risk bucket
- predicted higher-vol probability where available

Add:
- heuristic_severity
- hmm_v1_severity
- hmm_v2_severity
- hmm_v3_severity
- hmm_v3_1_severity
- v1_v3_severity_gap
- v1_v3_1_severity_gap
- v1_v2_severity_gap
- number_of_models_predicting_higher_vol
- number_of_models_predicting_low_risk
- model_severity_min
- model_severity_max
- model_severity_mean
- model_severity_std
- model_disagreement_count
- is_opposite_bucket_disagreement_present

HMMv4 estimator:
Initial baseline:
- GradientBoostingClassifier

Also run comparison estimators if simple:
- LogisticRegression
- RandomForestClassifier

But primary HMMv4 reported model should be:
- hmm_v4_path_aware_meta_gb

If multiple estimators are implemented, report them separately:
- hmm_v4_path_aware_logit
- hmm_v4_path_aware_rf
- hmm_v4_path_aware_gb

Do not call it deep learning yet.
Do not implement DNN in this slice.

HMMv4 outputs:
For each prediction:
- predicted_risk_bucket
- predicted_higher_vol_probability
- predicted_regime_probabilities if available
- target_horizon
- estimator_type
- training_rows
- feature_count
- fallback_used
- top_feature_importances
- path_feature_snapshot:
  - geometry_stress_score
  - geometry_stress_delta_5d
  - geometry_stress_curvature_5_10
  - vol_geometry_gap
  - vol_geometry_diverging
  - model_disagreement_count

Fallback:
If HMMv4 cannot train due to insufficient labeled rows:
- fallback to HMMv3.1 prediction
- set fallback_used = true
- include warning
But for the requested 10-year run, fallback rate should be near zero after the first warmup region.
If fallback rate > 10%, fail the run and explain why.

Backtest config:
Create:
configs/backtest/hmm_replay_10y_hmmv4.yaml

Must include:
start_date: 2016-01-01
end_date: latest_available
train_lookback_days: 2520
min_train_rows: 1260
horizons: [1, 2, 3, 5, 10]
models:
  - heuristic
  - hmm_v1_core
  - hmm_v2_core_plus_sector_corr
  - hmm_v3_core_plus_sector_geometry
  - hmm_v3_1_meta_blend
  - hmm_v4_path_aware_meta
retrain_each_date: true
random_state: 42
allow_partial_backtest: false
allow_silent_date_fallback: false
require_10y_replay: true
output_dir: reports/backtests/hmm_replay_10y/
artifact_dir: data/backtests/hmm_replay_10y/

CLI:
Add or support:

python -m src.backtest.hmm_replay.replay_runner \
  --config configs/backtest/hmm_replay_10y_hmmv4.yaml \
  --start-date 2016-01-01 \
  --end-date latest \
  --models heuristic,hmm_v1_core,hmm_v2_core_plus_sector_corr,hmm_v3_core_plus_sector_geometry,hmm_v3_1_meta_blend,hmm_v4_path_aware_meta \
  --horizons 1,2,3,5,10 \
  --require-10y \
  --no-partial-fallback

If Codex cannot implement "latest" as a CLI value, resolve latest_available from the feature store before running.

Report additions:
Create:
- replay_report_10y_hmmv4.md
- economic_score_summary_10y.csv
- hmmv4_path_feature_diagnostics.csv
- hmmv4_feature_importance.csv
- hmmv4_win_loss_by_condition.csv
- hmmv4_confusion_matrix.csv
- hmmv4_false_alarms.csv
- hmmv4_missed_risks.csv
- hmmv4_calibration_buckets.csv

Report sections:

1. 10-Year Backtest Preflight
Show:
- requested start date
- actual start date
- requested end date
- actual end date
- earliest feature-store date
- latest feature-store date
- train lookback rows
- min train rows
- whether full 10-year replay was honored

If actual_start_date > 2016-01-01:
mark report as FAILED, not successful.

2. Model Comparison
Compare all models by horizon:
- exact accuracy
- adjacent tolerant accuracy
- risk bucket accuracy
- missed risk rate
- false alarm rate
- Brier VIX Spike
- Brier Higher Vol Transition
- Brier RV Expansion
- directional VIX accuracy
- directional RV accuracy

3. HMMv4 Path-Aware Summary
Show:
- estimator type
- number of training samples
- average feature count
- fallback rate
- top 20 features
- best horizon
- worst horizon

4. HMMv4 vs HMMv3.1
Specifically compare:
- missed risk rate
- false alarm rate
- Brier Higher Vol Transition
- Brier VIX Spike
- risk bucket accuracy
- cases where v4 corrected v3.1
- cases where v4 made v3.1 worse

5. Path Diagnostics
Bucket by:
- geometry_stress_delta_5d
- geometry_stress_curvature_5_10
- vol_geometry_gap
- vol_geometry_diverging
- model_disagreement_count

For each bucket:
- count
- v4 win rate
- v4 loss rate
- missed risk rate
- false alarm rate

6. Calibration Buckets
For predicted_higher_vol_probability:
Buckets:
0.0-0.1
0.1-0.2
...
0.9-1.0

Show:
- forecast count
- avg predicted probability
- realized higher-vol frequency
- calibration error

7. Plain-English Interpretation
Generate deterministic summary:
- Did HMMv4 improve over HMMv3.1?
- Did path features help?
- Is HMMv4 reducing missed risks or mainly reducing false alarms?
- Is HMMv4 overfit-looking?
- Are feature importances dominated by model outputs or raw market features?
- Should HMMv4 remain research-only?

Important interpretation rule:
If HMMv4 improves in-sample but fails walk-forward scoring, say so.
Do not overstate.

Tests:
Add tests:
1. 10-year config refuses to run if feature store starts after 2016-01-01.
2. no silent fallback to 3 years.
3. preflight fails loudly with missing coverage.
4. HMMv4 path features use only data <= as_of_date.
5. HMMv4 targets use only future labels for training rows where future would be known before prediction date.
6. walk-forward training excludes prediction date and future rows.
7. fallback rate >10% fails 10-year run.
8. curvature features computed correctly.
9. divergence features computed correctly.
10. HMMv4 appears in model comparison report.
11. HMMv4 feature importance file is written.
12. HMMv4 calibration buckets are written.

Guardrails:
- Do not mutate existing HMMv1/v2/v3/v3.1 behavior.
- Do not promote HMMv4 to production.
- Do not call live IBKR during replay.
- Do not write production model artifacts.
- Do not use random train/test split.
- Do not silently reduce backtest range.
- Do not silently reduce train lookback.
- Do not hide preflight failures.

Implementation order:
1. Add strict 10-year preflight validation.
2. Add / verify 10-year feature store coverage.
3. Add HMMv4 path-aware feature builder.
4. Add walk-forward supervised target builder.
5. Add HMMv4 GradientBoostingClassifier adapter.
6. Add HMMv4 to replay model registry.
7. Add 10-year backtest config.
8. Add HMMv4 report sections.
9. Add tests.
10. Run full 10-year replay.
11. Produce report and CSV artifacts.

Expected outcome:
This slice should tell us whether HMMv4 learns useful nonlinear, path-dependent model trust behavior, or whether HMMv3.1’s simpler smooth blend is still sufficient.

Do not interpret one metric alone.
The primary decision metrics are:
- missed risk rate
- false alarm rate
- Brier Higher Vol Transition
- Brier VIX Spike
- risk bucket accuracy
- calibration bucket quality