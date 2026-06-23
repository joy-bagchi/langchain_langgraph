# HMM Replay Backtest Report

## Overall Model Comparison

_No rows_

## Prediction Distribution

_No rows_

## Outcome Distribution

_No rows_

## Confusion Matrix by Horizon

_No rows_

## Economic Score Summary

_No rows_

## False Alarms

_No rows_

## Missed Risks

_No rows_

## Recent As-of Date Comparison

| As Of      | Model                            | Predicted State          | T+1 Outcome   | T+2 Outcome   | T+3 Outcome   | Score   |
|:-----------|:---------------------------------|:-------------------------|:--------------|:--------------|:--------------|:--------|
| 2026-06-12 | heuristic                        | VOL_EXPANSION_TRANSITION |               |               |               |         |
| 2026-06-12 | hmm_v1_core                      | MID_VOL_CHOP             |               |               |               |         |
| 2026-06-12 | hmm_v2_core_plus_sector_corr     | MID_VOL_CHOP             |               |               |               |         |
| 2026-06-12 | hmm_v3_core_plus_sector_geometry | MID_VOL_CHOP             |               |               |               |         |
| 2026-06-12 | hmm_v3_1_meta_blend              | MID_VOL_CHOP             |               |               |               |         |
| 2026-06-12 | hmm_v4_path_aware_meta           | STABLE_LOW_VOL_TREND     |               |               |               |         |

## Model Disagreement Cases

| as_of_date   | hmm_v3       | other_model            | other_state              |
|:-------------|:-------------|:-----------------------|:-------------------------|
| 2026-06-12   | MID_VOL_CHOP | heuristic              | VOL_EXPANSION_TRANSITION |
| 2026-06-12   | MID_VOL_CHOP | hmm_v4_path_aware_meta | STABLE_LOW_VOL_TREND     |

## HMM v3 Special Section

Track whether geometry features improved false vol-expansion avoidance and mid-vol chop detection.

## Model Usefulness Summary

No economic summary rows were available.

## Disagreement Attribution

### Disagreement Summary Table

| comparison_model   |   horizon |   total_disagreements |   v3_win_rate |   v3_loss_rate |   tie_rate |   v3_bucket_win_rate |
|:-------------------|----------:|----------------------:|--------------:|---------------:|-----------:|---------------------:|
| heuristic          |         1 |                     1 |             0 |              0 |          1 |                    0 |
| heuristic          |         2 |                     1 |             0 |              0 |          1 |                    0 |
| heuristic          |         3 |                     1 |             0 |              0 |          1 |                    0 |
| heuristic          |         5 |                     1 |             0 |              0 |          1 |                    0 |
| heuristic          |        10 |                     1 |             0 |              0 |          1 |                    0 |

### Geometry Override Summary

| Type                        |   Count |   Success Rate | Notes                                         |
|:----------------------------|--------:|---------------:|:----------------------------------------------|
| v3_downgrade                |       1 |              0 | HMM v3 lower-severity than comparison model.  |
| v3_upgrade                  |       0 |              0 | HMM v3 higher-severity than comparison model. |
| opposite_bucket             |       1 |              0 | Highest-priority disagreements.               |
| same_bucket_different_state |       0 |              0 | Label variation inside same economic bucket.  |

### Top 20 Most Important Disagreements

| as_of_date   | comparison_model   | comparison_state         | hmm_v3_state   | realized_state_1d   | realized_state_2d   | realized_state_3d   | vix_change_pct_3d   |   avg_pairwise_corr_21d |   first_eigenvalue_share_21d |   effective_rank_21d |   log_det_corr_21d | v3_result   |
|:-------------|:-------------------|:-------------------------|:---------------|:--------------------|:--------------------|:--------------------|:--------------------|------------------------:|-----------------------------:|---------------------:|-------------------:|:------------|
| 2026-06-12   | heuristic          | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP   |                     |                     |                     |                     |                0.161761 |                     0.417111 |              5.09218 |           -8.57623 | tie         |

### Plain-English Interpretation

- HMM v3 changes labels but not enough to materially improve economic risk-bucket outcomes.

### Geometry Override Cases

| as_of_date   | comparison_model   | comparison_state         | hmm_v3_state   | disagreement_type   | is_opposite_bucket   |   severity_delta |   vix |   vvix |   vvix_vix_ratio |   vvix_vix_z_22d |   vix_vix3m_ratio |   vix9d_vix_ratio |   term_structure_slope |   realized_vol_5d |   realized_vol_21d |   realized_vol_trend | spy_return_1d   |   drawdown_21d |   trend_persistence_21d |   avg_pairwise_corr_21d |   first_eigenvalue_share_21d |   effective_rank_21d |   log_det_corr_21d | warning   | realized_state_1d   | realized_risk_bucket_1d   | vix_change_pct_1d   | vvix_change_pct_1d   | rv21_change_1d   | vix_spike_1d   | rv_expanded_1d   | risk_bucket_correct_hmm_v3_1d   | risk_bucket_correct_comparison_1d   | exact_correct_hmm_v3_1d   | exact_correct_comparison_1d   | directional_vol_correct_hmm_v3_1d   | directional_vol_correct_comparison_1d   | v3_won_1d   | v3_bucket_won_1d   | realized_state_2d   | realized_risk_bucket_2d   | spy_return_2d   | vix_change_pct_2d   | vvix_change_pct_2d   | rv21_change_2d   | vix_spike_2d   | rv_expanded_2d   | risk_bucket_correct_hmm_v3_2d   | risk_bucket_correct_comparison_2d   | exact_correct_hmm_v3_2d   | exact_correct_comparison_2d   | directional_vol_correct_hmm_v3_2d   | directional_vol_correct_comparison_2d   | v3_won_2d   | v3_bucket_won_2d   | realized_state_3d   | realized_risk_bucket_3d   | spy_return_3d   | vix_change_pct_3d   | vvix_change_pct_3d   | rv21_change_3d   | vix_spike_3d   | rv_expanded_3d   | risk_bucket_correct_hmm_v3_3d   | risk_bucket_correct_comparison_3d   | exact_correct_hmm_v3_3d   | exact_correct_comparison_3d   | directional_vol_correct_hmm_v3_3d   | directional_vol_correct_comparison_3d   | v3_won_3d   | v3_bucket_won_3d   | realized_state_5d   | realized_risk_bucket_5d   | spy_return_5d   | vix_change_pct_5d   | vvix_change_pct_5d   | rv21_change_5d   | vix_spike_5d   | rv_expanded_5d   | risk_bucket_correct_hmm_v3_5d   | risk_bucket_correct_comparison_5d   | exact_correct_hmm_v3_5d   | exact_correct_comparison_5d   | directional_vol_correct_hmm_v3_5d   | directional_vol_correct_comparison_5d   | v3_won_5d   | v3_bucket_won_5d   | realized_state_10d   | realized_risk_bucket_10d   | spy_return_10d   | vix_change_pct_10d   | vvix_change_pct_10d   | rv21_change_10d   | vix_spike_10d   | rv_expanded_10d   | risk_bucket_correct_hmm_v3_10d   | risk_bucket_correct_comparison_10d   | exact_correct_hmm_v3_10d   | exact_correct_comparison_10d   | directional_vol_correct_hmm_v3_10d   | directional_vol_correct_comparison_10d   | v3_won_10d   | v3_bucket_won_10d   | v3_result   |   geometry_stress_score |   downgrade_levels |
|:-------------|:-------------------|:-------------------------|:---------------|:--------------------|:---------------------|-----------------:|------:|-------:|-----------------:|-----------------:|------------------:|------------------:|-----------------------:|------------------:|-------------------:|---------------------:|:----------------|---------------:|------------------------:|------------------------:|-----------------------------:|---------------------:|-------------------:|:----------|:--------------------|:--------------------------|:--------------------|:---------------------|:-----------------|:---------------|:-----------------|:--------------------------------|:------------------------------------|:--------------------------|:------------------------------|:------------------------------------|:----------------------------------------|:------------|:-------------------|:--------------------|:--------------------------|:----------------|:--------------------|:---------------------|:-----------------|:---------------|:-----------------|:--------------------------------|:------------------------------------|:--------------------------|:------------------------------|:------------------------------------|:----------------------------------------|:------------|:-------------------|:--------------------|:--------------------------|:----------------|:--------------------|:---------------------|:-----------------|:---------------|:-----------------|:--------------------------------|:------------------------------------|:--------------------------|:------------------------------|:------------------------------------|:----------------------------------------|:------------|:-------------------|:--------------------|:--------------------------|:----------------|:--------------------|:---------------------|:-----------------|:---------------|:-----------------|:--------------------------------|:------------------------------------|:--------------------------|:------------------------------|:------------------------------------|:----------------------------------------|:------------|:-------------------|:---------------------|:---------------------------|:-----------------|:---------------------|:----------------------|:------------------|:----------------|:------------------|:---------------------------------|:-------------------------------------|:---------------------------|:-------------------------------|:-------------------------------------|:-----------------------------------------|:-------------|:--------------------|:------------|------------------------:|-------------------:|
| 2026-06-12   | heuristic          | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP   | v3_downgrade        | True                 |               -1 | 17.68 |  93.82 |          5.30656 |       -0.0916598 |          0.862019 |          0.976244 |                   2.83 |           18.9995 |            15.0315 |              3.96806 |                 |      0.0234607 |                0.619048 |                0.161761 |                     0.417111 |              5.09218 |           -8.57623 |           |                     | LOW_RISK                  |                     |                      |                  |                |                  | False                           | False                               | False                     | False                         |                                     |                                         | tie         | tie                |                     | LOW_RISK                  |                 |                     |                      |                  |                |                  | False                           | False                               | False                     | False                         |                                     |                                         | tie         | tie                |                     | LOW_RISK                  |                 |                     |                      |                  |                |                  | False                           | False                               | False                     | False                         |                                     |                                         | tie         | tie                |                     | LOW_RISK                  |                 |                     |                      |                  |                |                  | False                           | False                               | False                     | False                         |                                     |                                         | tie         | tie                |                      | LOW_RISK                   |                  |                      |                       |                   |                 |                   | False                            | False                                | False                      | False                          |                                      |                                          | tie          | tie                 | tie         |                0.364484 |                  0 |

### Geometry False Suppression Cases

_No rows_

### Geometry False Suppression Analysis

| segment   | metric     | category   |   value |
|:----------|:-----------|:-----------|--------:|
| overall   | case_count |            |       0 |

### Geometry Success Cases

_No rows_

## Geometry Smooth Modifier

| as_of_date   |   core_vol_risk_score |   geometry_stress_score |   final_risk_score | core_vol_state   | final_regime   |   downgrade_levels | downgrade_cap_applied   |
|:-------------|----------------------:|------------------------:|-------------------:|:-----------------|:---------------|-------------------:|:------------------------|
| 2026-06-12   |              0.332347 |                0.364484 |           0.340382 | MID_VOL_CHOP     | MID_VOL_CHOP   |                  0 | False                   |

## Path-Aware Meta Learner

| as_of_date   |   target_horizon |   training_row_count | path_aware_estimator   | fallback_used   | feature_families_used                                                    | predicted_risk_bucket   |   geometry_stress_score |   geometry_stress_delta_5d |   geometry_stress_curvature_5_10 |   vol_geometry_gap |   vol_geometry_diverging | top_feature_importances   |
|:-------------|-----------------:|---------------------:|:-----------------------|:----------------|:-------------------------------------------------------------------------|:------------------------|------------------------:|---------------------------:|---------------------------------:|-------------------:|-------------------------:|:--------------------------|
| 2026-06-12   |                3 |                 2456 | constant_single_class  | False           | curvature | deltas_slopes | divergence | ensemble | levels | persistence | LOW_RISK                |                0.364484 |                   0.273016 |                        0.0667857 |           0.130288 |                        0 |                           |

## Path Feature Diagnostics

_No rows_

## Diagnostics

| as_of_date   | model_name                       | converged   |   training_row_count | training_end_date   | warnings                                                                                                                                                                                                                                                            |
|:-------------|:---------------------------------|:------------|---------------------:|:--------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-06-12   | heuristic                        | True        |                 2520 | 2026-06-12          |                                                                                                                                                                                                                                                                     |
| 2026-06-12   | hmm_v1_core                      | True        |                 2520 | 2026-06-12          |                                                                                                                                                                                                                                                                     |
| 2026-06-12   | hmm_v2_core_plus_sector_corr     | True        |                 2520 | 2026-06-12          |                                                                                                                                                                                                                                                                     |
| 2026-06-12   | hmm_v3_core_plus_sector_geometry | True        |                 2520 | 2026-06-12          |                                                                                                                                                                                                                                                                     |
| 2026-06-12   | hmm_v3_1_meta_blend              | True        |                 2520 | 2026-06-12          | Geometry stress uses lookback=252. | avg_corr_stress=0.03, eigen_stress=0.32, effective_rank_stress=0.59, log_det_stress=0.74. | Final risk score uses weighted blend: core=0.75, geometry=0.25. | Core state=MID_VOL_CHOP (confidence=0.99), geometry_stress=0.36. |
| 2026-06-12   | hmm_v4_path_aware_meta           |             |                 2456 |                     | Insufficient geometry history for stress scoring; using neutral geometry stress 0.50. | HMM v4 path-aware meta model training labels collapsed to one class; using constant classifier for this as-of date instead of fallback.                                     |
