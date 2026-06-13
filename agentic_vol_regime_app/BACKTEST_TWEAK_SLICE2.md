Backtest Slice 2: Disagreement Attribution Report

Context:
The HMM replay backtest now compares Heuristic, HMM v1, HMM v2, and HMM v3. HMM v3 is not clearly dominating aggregate metrics yet, but it appears to behave differently. It may be acting as a false-alarm filter by downgrading Vol Expansion calls when sector geometry says market structure is still intact.

Goal:
Create a Disagreement Attribution Report that identifies when HMM v3 disagrees with HMM v1/v2/heuristic, explains which features likely drove the disagreement, and shows whether HMM v3 was right or wrong over T+1/T+2/T+3.

Do not change model training.
Do not change HMM features.
Do not change Daily Belief Report.
Only enhance replay scoring/reporting.

Primary question:
When geometry changes the decision, is it usually right?

Models:
- heuristic
- hmm_v1_core
- hmm_v2_core_plus_sector_corr
- hmm_v3_core_plus_sector_geometry

Horizons:
- 1
- 2
- 3

Required new outputs:
- disagreement_attribution.csv
- disagreement_summary.csv
- geometry_override_cases.csv
- geometry_false_suppression_cases.csv
- geometry_success_cases.csv
- updated replay_report.md section: "Disagreement Attribution"

Core idea:
Treat model disagreements as first-class research objects.

A disagreement case exists when, for the same as_of_date:
- HMM v3 predicted state differs from HMM v1, OR
- HMM v3 predicted state differs from HMM v2, OR
- HMM v3 predicted state differs from heuristic

Regime severity mapping:
STABLE_LOW_VOL = 0
MID_VOL_CHOP = 1
VOL_EXPANSION_TRANSITION = 2
HIGH_VOL_RISK_OFF = 3

If code has slightly different names, explicitly map them.

Disagreement types:

1. v3_downgrade
HMM v3 predicts a lower-severity regime than comparison model.

Example:
HMM v1 = VOL_EXPANSION_TRANSITION
HMM v3 = MID_VOL_CHOP

Interpretation:
Geometry or sector features may be suppressing a vol-expansion warning.

2. v3_upgrade
HMM v3 predicts a higher-severity regime than comparison model.

Example:
HMM v1 = MID_VOL_CHOP
HMM v3 = VOL_EXPANSION_TRANSITION

Interpretation:
Geometry or sector features may be detecting internal market deterioration.

3. v3_same_bucket_different_state
Both predictions are in same broad risk bucket, but different exact state.

Example:
HMM v1 = STABLE_LOW_VOL
HMM v3 = MID_VOL_CHOP

Risk buckets:
LOW_RISK:
- STABLE_LOW_VOL
- MID_VOL_CHOP

HIGHER_VOL_RISK:
- VOL_EXPANSION_TRANSITION
- HIGH_VOL_RISK_OFF

4. v3_opposite_bucket
HMM v3 and comparison model are in different risk buckets.

Example:
HMM v1 = VOL_EXPANSION_TRANSITION
HMM v3 = STABLE_LOW_VOL

This is highest priority.

Feature snapshot to include for every disagreement:
- as_of_date
- comparison_model
- comparison_state
- hmm_v3_state
- disagreement_type
- severity_delta = v3_severity - comparison_severity
- vix
- vvix
- vvix_vix_ratio
- vvix_vix_z_22d
- vix_vix3m_ratio
- vix9d_vix_ratio
- term_structure_slope
- realized_vol_5d
- realized_vol_21d
- realized_vol_trend
- spy_return_1d
- drawdown_21d
- trend_persistence_21d
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d

If any feature is missing, leave blank and add warning column.

Outcome fields for each horizon:
For T+1, T+2, T+3 include:
- realized_state_h
- realized_risk_bucket_h
- spy_return_h
- vix_change_pct_h
- vvix_change_pct_h
- rv21_change_h
- vix_spike_h
- rv_expanded_h
- risk_bucket_correct_hmm_v3
- risk_bucket_correct_comparison
- exact_correct_hmm_v3
- exact_correct_comparison
- directional_vol_correct_hmm_v3
- directional_vol_correct_comparison

Outcome judgment:
For each disagreement and horizon, compute:

v3_won_h:
true if HMM v3 was closer to realized state than comparison model.

Use severity distance:
v3_distance = abs(v3_severity - realized_severity)
comparison_distance = abs(comparison_severity - realized_severity)

if v3_distance < comparison_distance:
  v3_won_h = true
elif v3_distance > comparison_distance:
  v3_won_h = false
else:
  tie

Also compute:
v3_bucket_won_h:
true if HMM v3 got risk bucket correct and comparison did not.

false if comparison got risk bucket correct and HMM v3 did not.

tie otherwise.

Geometry override cases:
Create geometry_override_cases.csv where:
- disagreement_type is v3_downgrade or v3_upgrade
- severity_delta absolute value >= 1
- HMM v3 differs from HMM v1 or heuristic

Geometry success cases:
Create geometry_success_cases.csv where:
- HMM v3 disagreed with comparison model
- HMM v3 won on at least one horizon
- Include which horizon and why

Geometry false suppression cases:
Create geometry_false_suppression_cases.csv where:
- disagreement_type = v3_downgrade
- comparison model predicted higher-vol risk
- HMM v3 predicted lower-vol risk
- realized outcome became higher-vol risk
- This means HMM v3 suppressed a warning incorrectly

Geometry false alarm filter cases:
Create geometry_success_cases subset or flag:
- disagreement_type = v3_downgrade
- comparison predicted higher-vol risk
- HMM v3 predicted lower-vol risk
- realized outcome stayed low-risk
- VIX fell or stayed flat
- This is evidence HMM v3 filtered a false alarm

Summary metrics:
For each comparison model and horizon compute:
- total_disagreements
- v3_win_rate
- v3_loss_rate
- tie_rate
- v3_bucket_win_rate
- v3_bucket_loss_rate
- downgrade_count
- upgrade_count
- downgrade_success_rate
- upgrade_success_rate
- false_suppression_rate
- false_alarm_filter_success_rate

Add these to disagreement_summary.csv.

Replay report additions:

Add section:
"Disagreement Attribution"

Include:

1. Disagreement Summary Table

| Comparison Model | Horizon | Disagreements | V3 Win Rate | V3 Loss Rate | Tie Rate | V3 Bucket Win Rate |
|---|---:|---:|---:|---:|---:|---:|

2. Geometry Override Summary

| Type | Count | Success Rate | Notes |
|---|---:|---:|---|

Types:
- v3_downgrade
- v3_upgrade
- opposite_bucket
- same_bucket_different_state

3. Top 20 Most Important Disagreements

Sort by:
- opposite_bucket first
- largest abs(severity_delta)
- largest abs(vix_change_pct_3d)
- largest abs(rv21_change_3d)

Columns:
- as_of_date
- comparison_model
- comparison_state
- hmm_v3_state
- T+1 realized
- T+2 realized
- T+3 realized
- vix_change_pct_3d
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d
- v3_result

4. Plain-English Interpretation

Generate a small deterministic summary, not LLM-based.

Examples:

If v3_downgrade success rate > 60%:
"HMM v3 appears to be usefully suppressing false vol-expansion warnings."

If false_suppression_rate > 30%:
"HMM v3 may be too conservative and may under-warn before higher-vol transitions."

If v3_upgrade success rate > 60%:
"HMM v3 appears to detect internal market deterioration before the core HMM."

If most disagreements are ties:
"HMM v3 changes labels but not enough to materially improve economic risk-bucket outcomes."

Important:
This report should help answer:
- Is HMM v3 just conservative?
- Does HMM v3 reduce false alarms?
- Does HMM v3 miss important risk transitions?
- Do geometry features add useful independent signal?
- When HMM v3 disagrees, should we trust it?

Tests:
Add tests for:
1. disagreement detection when HMM v3 differs from HMM v1
2. v3_downgrade classification
3. v3_upgrade classification
4. opposite risk-bucket detection
5. severity-distance win/loss/tie logic
6. bucket win/loss/tie logic
7. false suppression detection
8. false alarm filter success detection
9. disagreement summary aggregation
10. report renders Disagreement Attribution section

Implementation order:
1. Add regime severity and risk bucket helpers if not already present.
2. Load prediction records grouped by as_of_date.
3. Join HMM v3 predictions against heuristic/HMM v1/HMM v2 predictions.
4. Join outcomes for horizons 1/2/3.
5. Compute disagreement rows.
6. Compute v3 win/loss/tie.
7. Write disagreement_attribution.csv.
8. Write disagreement_summary.csv.
9. Write geometry_* case files.
10. Update replay_report.md.
11. Add tests.

Guardrails:
- Do not retrain models.
- Do not change prediction records.
- Do not mutate production artifacts.
- Do not call IBKR.
- This is reporting/scoring only.