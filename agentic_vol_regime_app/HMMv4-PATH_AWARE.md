Build HMMv4 Path-Aware Meta Model

Context:
HMMv3.1 Meta-Blend introduced a smooth geometry_stress_score and blended it with HMMv1 core volatility risk. This improved behavior versus raw HMMv3, but it is still mostly a static point-in-time blend.

Problem:
Volatility is heteroskedastic and path-dependent. The same VIX/VVIX/geometry levels can lead to different future outcomes depending on the path taken to get there:
- Is geometry stress rising or falling?
- Is it accelerating?
- How long has it been elevated?
- Is vol confirming geometry?
- Are vol and geometry diverging?

Goal:
Add a new experimental path-aware model:
- hmm_v4_path_aware_meta

Do not modify HMMv1, HMMv2, HMMv3, or HMMv3.1 behavior.
Do not remove existing reports.
Do not change production policy.
Add v4 as a separate experimental model available in backtest/replay.

Core design:
HMMv4 should learn from a sliding window / path-aware feature representation rather than only same-day scalar values.

It should use:
1. Current market feature levels.
2. Feature deltas over multiple windows.
3. Feature slopes.
4. Feature acceleration / curvature.
5. Persistence / days-above-threshold features.
6. Existing ensemble model outputs.

Initial implementation:
Use tabular supervised meta-learning, not deep learning yet.

Preferred first models:
- GradientBoostingClassifier
- RandomForestClassifier
- LogisticRegression baseline

Optional later:
- MLPClassifier / PyTorch small DNN only after tree/logistic baselines work.

New files:
- src/features/path_aware_features.py
- src/agents/hmm_v4_path_aware_meta_agent.py
- src/backtest/hmm_replay/path_aware_dataset.py
- configs/models/hmm_v4_path_aware_meta.yaml
- tests/test_path_aware_features.py
- tests/test_hmm_v4_path_aware_meta.py

Use existing HMMv3.1 geometry stress code:
If GeometryStressAgent already exists, reuse it.
Do not duplicate logic.

Geometry inputs:
Use existing features:
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d

Geometry stress components:
Use or expose existing components:
- avg_corr_stress
- eigen_stress
- effective_rank_stress
- log_det_stress
- geometry_stress_score

If these are not currently persisted in prediction records or feature store, persist them.

Path-aware geometry features:
For each as_of_date, using only data <= as_of_date, compute:

1. Level features:
- geometry_stress_score
- avg_corr_stress
- eigen_stress
- effective_rank_stress
- log_det_stress
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d

2. Delta features:
For windows [1, 3, 5, 10, 21]:
- geometry_stress_delta_{w}d
- avg_corr_stress_delta_{w}d
- eigen_stress_delta_{w}d
- effective_rank_stress_delta_{w}d
- log_det_stress_delta_{w}d
- avg_pairwise_corr_delta_{w}d
- first_eigenvalue_share_delta_{w}d
- effective_rank_delta_{w}d
- log_det_corr_delta_{w}d

3. Slope features:
For windows [3, 5, 10, 21]:
- geometry_stress_slope_{w}d = geometry_stress_delta_{w}d / w
- eigen_stress_slope_{w}d
- effective_rank_stress_slope_{w}d
- log_det_stress_slope_{w}d

4. Curvature / acceleration features:
Define:
curvature_5_10 =
  (geometry_stress_score[t] - geometry_stress_score[t-5]) / 5
  -
  (geometry_stress_score[t-5] - geometry_stress_score[t-10]) / 5

Add:
- geometry_stress_curvature_5_10
- geometry_stress_curvature_10_21
- eigen_stress_curvature_5_10
- effective_rank_stress_curvature_5_10
- log_det_stress_curvature_5_10

5. Persistence features:
For trailing windows [5, 10, 21, 63]:
- geometry_days_above_0_55_{w}d
- geometry_days_above_0_70_{w}d
- geometry_days_below_0_30_{w}d
- geometry_mean_{w}d
- geometry_max_{w}d
- geometry_min_{w}d
- geometry_std_{w}d

6. Divergence features:
These are very important.

Compute:
- core_vol_risk_score from HMMv1 probabilities
- geometry_stress_score
- vol_geometry_gap = core_vol_risk_score - geometry_stress_score
- vol_geometry_gap_delta_5d
- vol_geometry_gap_delta_10d
- vol_geometry_gap_abs
- vol_geometry_confirming = 1 if both core_vol_risk_score and geometry_stress_score are above 0.55
- vol_geometry_diverging = 1 if core_vol_risk_score > 0.60 and geometry_stress_score < 0.35, or vice versa

7. Vol path features:
For windows [1, 3, 5, 10, 21]:
- vix_delta_{w}d
- vvix_delta_{w}d
- vvix_vix_ratio_delta_{w}d
- vix_vix3m_ratio_delta_{w}d
- vix9d_vix_ratio_delta_{w}d
- realized_vol_21d_delta_{w}d
- drawdown_21d_delta_{w}d
- trend_persistence_21d_delta_{w}d

8. Ensemble output features:
For each model:
- heuristic top state severity
- hmm_v1 state probabilities
- hmm_v2 state probabilities
- hmm_v3 state probabilities
- hmm_v3_1 final_risk_score
- hmm_v3_1 final_regime severity
- hmm_v3_1 geometry_stress_score
- hmm_v3_1 downgrade_levels
- hmm_v3_1 downgrade_cap_applied

Add disagreement features:
- v1_v3_severity_gap
- v1_v3_1_severity_gap
- v1_v2_severity_gap
- number_of_models_predicting_higher_vol
- number_of_models_predicting_low_risk
- max_model_severity
- min_model_severity
- model_severity_dispersion

Targets:
Train supervised model to predict realized future regime or risk bucket.

Create targets for horizons:
- 1d
- 2d
- 3d
- optionally 5d later

Primary target:
- realized_risk_bucket_h
  LOW_RISK vs HIGHER_VOL_RISK

Secondary target:
- realized_state_h
  4-class regime label

Tertiary binary targets:
- vix_spike_h
- rv_expanded_h
- higher_vol_transition_h

Initial v4 should optimize primary target first:
LOW_RISK vs HIGHER_VOL_RISK.

Training method:
Use walk-forward training only.

For each as_of_date in replay:
- training rows must be < as_of_date
- no future outcomes beyond as_of_date may be visible
- prediction row is as_of_date
- score against future horizon

Do not use random train/test split.

Add config:
configs/models/hmm_v4_path_aware_meta.yaml

Fields:
model_name: hmm_v4_path_aware_meta
model_type: supervised_meta_learner
enabled: true
base_estimator: gradient_boosting
target: realized_risk_bucket
horizon: 3
min_training_rows: 250
walk_forward_train_lookback_days: 756
feature_windows: [1, 3, 5, 10, 21, 63]
geometry_stress_lookback: 252
random_state: 42
fallback_model: hmm_v3_1_meta_blend

Fallback:
If fewer than min_training_rows exist, return:
- warning
- fallback to hmm_v3_1_meta_blend prediction

Output schema:
{
  "model_name": "hmm_v4_path_aware_meta",
  "as_of_date": "...",
  "target_horizon": 3,
  "predicted_risk_bucket": "LOW_RISK" or "HIGHER_VOL_RISK",
  "predicted_regime_probabilities": {
    "STABLE_LOW_VOL": 0.0,
    "MID_VOL_CHOP": 0.0,
    "VOL_EXPANSION_TRANSITION": 0.0,
    "HIGH_VOL_RISK_OFF": 0.0
  },
  "predicted_higher_vol_probability": 0.0,
  "model_trust_weights": {
    "heuristic": 0.0,
    "hmm_v1_core": 0.0,
    "hmm_v2_core_plus_sector_corr": 0.0,
    "hmm_v3_core_plus_sector_geometry": 0.0,
    "hmm_v3_1_meta_blend": 0.0
  },
  "path_features": {
    "geometry_stress_score": 0.0,
    "geometry_stress_delta_5d": 0.0,
    "geometry_stress_curvature_5_10": 0.0,
    "vol_geometry_gap": 0.0,
    "vol_geometry_diverging": 0
  },
  "top_feature_importances": [],
  "warnings": []
}

Important:
If the estimator does not naturally produce model_trust_weights, approximate them using:
- feature importance grouped by base model output feature family
- or leave as null with warning
Do not invent precise trust weights unless implemented.

Backtest integration:
Add hmm_v4_path_aware_meta to replay model list.

Run side-by-side:
- heuristic
- hmm_v1_core
- hmm_v2_core_plus_sector_corr
- hmm_v3_core_plus_sector_geometry
- hmm_v3_1_meta_blend
- hmm_v4_path_aware_meta

Report additions:
Add section:
"Path-Aware Meta Learner"

Include:
- target horizon
- training rows per replay date
- estimator type
- fallback count
- feature families used
- top feature importances
- risk bucket accuracy
- missed risk rate
- false alarm rate
- Brier Higher Vol Transition
- Brier VIX Spike
- comparison vs HMMv3.1

Add section:
"Path Feature Diagnostics"

Include:
- avg geometry_stress_delta_5d for wins vs losses
- avg geometry_curvature_5_10 for wins vs losses
- avg vol_geometry_gap for wins vs losses
- success rate when geometry accelerating
- success rate when geometry decelerating
- success rate when vol and geometry diverge
- success rate when vol and geometry confirm

Tests:
1. path-aware features never use future rows.
2. delta features are computed correctly.
3. curvature features are computed correctly.
4. persistence features are computed correctly.
5. vol_geometry_gap is computed correctly.
6. insufficient history triggers fallback.
7. walk-forward training excludes as_of_date outcome.
8. v4 can run in replay without mutating v1/v2/v3/v3.1.
9. report renders path-aware sections.
10. feature importance groups are computed if estimator supports them.

Guardrails:
- Do not use live IBKR during replay.
- Do not write to production model artifacts.
- Do not auto-promote v4 into live policy.
- Do not replace HMMv3.1.
- Do not implement DNN yet unless explicitly requested.
- Keep this experimental.

Implementation order:
1. Persist/expose geometry_stress_score and components from HMMv3.1.
2. Build path_aware_features.py.
3. Add target builder for realized risk bucket and realized state.
4. Build walk-forward dataset builder.
5. Train GradientBoostingClassifier baseline.
6. Add v4 replay model adapter.
7. Add reports and diagnostics.
8. Add tests.
9. Run same frozen feature store/date range as previous backtests.
