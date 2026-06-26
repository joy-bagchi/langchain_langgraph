Backtest Tweak Slice: Add Economic Forecast Diagnostics for HMM Replay

Context:
The current HMM replay backtest works and produces Brier scores, accuracy, and diagnostics. However, the current "accuracy" metric appears to compare predicted regime label directly against future realized regime label. That may understate model usefulness because neighboring regimes such as STABLE_LOW_VOL and MID_VOL_CHOP can both be economically correct if VIX falls and SPY rises.

Goal:
Improve the HMM replay report so it evaluates whether the model is economically useful for the overwrite strategy, not just whether it exactly predicts the next regime label.

Do not change model training.
Do not change HMM v1/v2/v3 configs.
Do not change Daily Belief Report.
Only modify the replay/backtest scoring and report output.

Primary outputs to add:
1. Prediction distribution
2. Outcome distribution
3. Confusion matrix
4. Adjacent-regime tolerant accuracy
5. Directional volatility accuracy
6. Risk-bucket accuracy
7. False alarm / missed risk report
8. Model usefulness summary

Models:
Support all replay models, but this slice can be tested with:
- hmm_v3_core_plus_sector_geometry

Horizons:
Support existing horizons:
- T+1
- T+2
- T+3

Regime ordering:
Define ordered regime severity:

STABLE_LOW_VOL = 0
MID_VOL_CHOP = 1
VOL_EXPANSION_TRANSITION = 2
HIGH_VOL_RISK_OFF = 3

If current code has slightly different enum names, map them explicitly.

1. Prediction Distribution

For each model and horizon, count predicted top states:

Example output:

| Model | Horizon | Predicted State | Count | Percent |
|---|---:|---|---:|---:|
| hmm_v3 | 1 | STABLE_LOW_VOL | 28 | 72% |
| hmm_v3 | 1 | MID_VOL_CHOP | 7 | 18% |
| hmm_v3 | 1 | VOL_EXPANSION_TRANSITION | 4 | 10% |
| hmm_v3 | 1 | HIGH_VOL_RISK_OFF | 0 | 0% |

Purpose:
Detect if model is just predicting one state all the time.

2. Outcome Distribution

For each horizon, count realized outcome states:

| Horizon | Realized State | Count | Percent |
|---:|---|---:|---:|

Purpose:
Compare model prediction distribution against actual state distribution.

3. Confusion Matrix

For each horizon, produce confusion matrix:

Rows = predicted state
Columns = realized future state

Example:

| Predicted \ Actual | STABLE | CHOP | EXPANSION | RISK_OFF |
|---|---:|---:|---:|---:|

Add both count and row-normalized percent if easy.

Purpose:
Understand which mistakes the model makes.

4. Adjacent-Regime Tolerant Accuracy

Exact accuracy is too harsh for neighboring states.

Define severity index:
STABLE_LOW_VOL = 0
MID_VOL_CHOP = 1
VOL_EXPANSION_TRANSITION = 2
HIGH_VOL_RISK_OFF = 3

exact_correct:
predicted_index == actual_index

adjacent_correct:
abs(predicted_index - actual_index) <= 1

severe_miss:
abs(predicted_index - actual_index) >= 2

Report:
- exact_accuracy
- adjacent_tolerant_accuracy
- severe_miss_rate

Purpose:
If model predicts STABLE and actual is CHOP, that should not be treated as a severe economic failure.
If model predicts STABLE and actual is HIGH_VOL_RISK_OFF, that is a severe failure.

5. Directional Volatility Accuracy

Add a more economically meaningful scoring layer.

For each as_of_date and horizon, compute observed future direction:

vix_direction:
- UP if VIX_h > VIX_asof
- DOWN if VIX_h < VIX_asof
- FLAT if absolute percent change < configurable threshold, default 1%

vvix_direction:
- UP/DOWN/FLAT using same logic

rv21_direction:
- UP if RV21_h > RV21_asof
- DOWN if RV21_h < RV21_asof
- FLAT if change below threshold

spy_direction:
- UP if SPY_h > SPY_asof
- DOWN if SPY_h < SPY_asof
- FLAT if abs return below threshold, default 0.15%

Map predicted regime to expected vol direction:

Predicted STABLE_LOW_VOL:
- expected VIX direction: DOWN_OR_FLAT
- expected VVIX direction: DOWN_OR_FLAT
- expected RV direction: DOWN_OR_FLAT

Predicted MID_VOL_CHOP:
- expected VIX direction: FLAT_OR_DOWN
- expected VVIX direction: FLAT_OR_DOWN
- expected RV direction: FLAT_OR_MIXED

Predicted VOL_EXPANSION_TRANSITION:
- expected VIX direction: UP_OR_FLAT
- expected VVIX direction: UP_OR_FLAT
- expected RV direction: UP_OR_FLAT

Predicted HIGH_VOL_RISK_OFF:
- expected VIX direction: UP_OR_ELEVATED
- expected VVIX direction: UP_OR_ELEVATED
- expected RV direction: UP_OR_ELEVATED

Implement:
directional_vix_correct
directional_vvix_correct
directional_rv_correct

Report by model/horizon:
- vix_directional_accuracy
- vvix_directional_accuracy
- rv_directional_accuracy
- combined_vol_directional_accuracy

combined_vol_directional_accuracy:
average of available directional correctness flags.

Purpose:
This directly tests whether HMM state prediction correctly anticipated vol easing vs vol worsening.

6. Risk-Bucket Accuracy

Map regimes into economic risk buckets:

LOW_RISK:
- STABLE_LOW_VOL
- MID_VOL_CHOP

HIGHER_VOL_RISK:
- VOL_EXPANSION_TRANSITION
- HIGH_VOL_RISK_OFF

For each horizon:

predicted_risk_bucket = LOW_RISK or HIGHER_VOL_RISK
actual_risk_bucket = LOW_RISK or HIGHER_VOL_RISK

Report:
- risk_bucket_accuracy
- false_alarm_rate
- missed_risk_rate

Definitions:
false_alarm:
predicted HIGHER_VOL_RISK but actual LOW_RISK

missed_risk:
predicted LOW_RISK but actual HIGHER_VOL_RISK

Purpose:
This is closer to overwrite decision quality than exact state accuracy.

7. False Alarm / Missed Risk Report

Create a section listing dates where the model was meaningfully wrong.

False alarms:
- predicted VOL_EXPANSION_TRANSITION or HIGH_VOL_RISK_OFF
- actual state stayed STABLE_LOW_VOL or MID_VOL_CHOP
- VIX fell or stayed flat

Missed risks:
- predicted STABLE_LOW_VOL or MID_VOL_CHOP
- actual moved to VOL_EXPANSION_TRANSITION or HIGH_VOL_RISK_OFF
- VIX rose materially or RV expanded

For each case include:
- as_of_date
- horizon
- predicted_state
- actual_state
- VIX change %
- VVIX change %
- RV21 change
- SPY return %
- feature snapshot if available:
  - VIX
  - VVIX/VIX
  - VIX term structure
  - avg_pairwise_corr_21d
  - first_eigenvalue_share_21d
  - effective_rank_21d
  - log_det_corr_21d

Purpose:
These are the highest-learning-value cases.

8. Model Usefulness Summary

Add a final plain-English summary section to replay_report.md:

Include:
- Is exact accuracy low but adjacent accuracy high?
- Is Brier VIX Spike good?
- Is the model mostly avoiding false vol-expansion calls?
- Is the model missing severe risk transitions?
- Does HMM v3 appear to function as a false-alarm filter?
- Does HMM v3 predict STABLE too often?

Example language:
"HMM v3 exact state accuracy is low, but adjacent-regime tolerant accuracy is materially higher, suggesting most errors are neighboring-regime errors rather than severe misses."

or:
"HMM v3 has low Brier VIX Spike but high missed-risk rate, suggesting it is conservative and good at predicting non-spike periods but may under-warn before high-vol transitions."

Data requirements:
Use existing prediction_records.jsonl, outcome_records.jsonl, scored_records.jsonl if available.
If needed, recompute from replay artifacts.
Do not rerun models unless necessary.
This should be mostly a scoring/reporting enhancement.

Add output files:
- prediction_distribution.csv
- outcome_distribution.csv
- confusion_matrix_by_horizon.csv
- economic_score_summary.csv
- false_alarms.csv
- missed_risks.csv
- replay_report.md updated with new sections

Tests:
Add or update tests for:

1. severity mapping works.
2. adjacent-regime tolerant accuracy counts neighboring regimes as correct.
3. severe miss rate identifies jumps of >=2 severity levels.
4. risk bucket mapping works.
5. false alarm detection works.
6. missed risk detection works.
7. VIX directional accuracy works.
8. Report renders new sections.
9. Existing Brier metrics remain unchanged.
10. No model retraining is triggered by this reporting-only slice.

Implementation order:
1. Add regime severity mapping.
2. Add risk bucket mapping.
3. Add prediction/outcome distribution functions.
4. Add confusion matrix function.
5. Add adjacent accuracy and severe miss metrics.
6. Add directional volatility scoring.
7. Add false alarm/missed risk extraction.
8. Add CSV outputs.
9. Update replay_report.md.
10. Add tests.

Important:
Do not interpret exact regime accuracy as the primary success metric anymore.
The most important metrics for this app are:
- Brier VIX Spike
- Brier Higher Vol Transition
- risk_bucket_accuracy
- missed_risk_rate
- false_alarm_rate
- adjacent_tolerant_accuracy
- directional VIX/RV accuracy