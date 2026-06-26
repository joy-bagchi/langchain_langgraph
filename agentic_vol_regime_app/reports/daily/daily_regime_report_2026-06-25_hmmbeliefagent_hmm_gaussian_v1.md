# Daily Volatility Regime Report

Date: 2026-06-25T03:08:40.894705+00:00

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.00 |
| Mid-Vol Chop | 0.00 |
| Vol Expansion Transition | 1.00 |
| High-Vol Risk-Off | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.63 | Current implied-vol level |
| VVIX | 97.38 | Vol-of-vol state |
| VVIX/VIX | 5.23 | Convexity stress ratio |
| VVIX/VIX z-score | -0.25 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.079 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WARNING`

Headline: Volatility expansion risk is rising with confirming signals

Drivers:
- realized volatility acceleration
- term structure flattening

## Policy Recommendation

Recommended action: `MEDIUM_OVERWRITE`

Rationale:
- Transition risk is elevated and argues for additional premium capture.

Risk notes:
- None

Overwrite implementation:
- Suggested call overwrite strike: `743.00`
- Suggested DTE: `3`
- Notes: HMM transition risk is elevated, so the overwrite duration was shortened.

## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v1 Core`

Model converged: `True`

Trained and active.

Top HMM state: `VOL_EXPANSION`

Current-state expected duration: `6.41` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.00 |
| MID_VOL_CHOP | 0.00 |
| VOL_EXPANSION | 1.00 |
| HIGH_VOL_STRESS | 0.00 |

### HMM Persistence

- Current state persists 5d: `0.50`
- Current state persists 10d: `0.36`
- Current state persists 21d: `0.28`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.59`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.45`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.36`

Warnings:
- Missing sector history for: XLK, XLF, XLE, XLY, XLP, XLI, XLB, XLV, XLU, XLRE
- Need at least 9 sector return series for sector-correlation features.
- Insufficient fully populated feature rows for HMM training.
- Fell back to HMM v1 because HMM v3 Core + Geometry sector/geometry data was incomplete.


## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | VOL_EXPANSION | 0.90 | 6.41 | 0.10 | MEDIUM_OVERWRITE |
| HMM v2 Core + Sector Corr | VOL_EXPANSION | 0.90 | 6.41 | 0.10 | MEDIUM_OVERWRITE |
| HMM v3 Core + Geometry | VOL_EXPANSION | 0.90 | 6.41 | 0.10 | MEDIUM_OVERWRITE |
| HMM v3.1 Meta-Blend | VOL_EXPANSION | 0.90 | 6.41 | 0.10 | MEDIUM_OVERWRITE |


## Model Confidence

Confidence: 0.90

Uncertainty / entropy: 0.00

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
