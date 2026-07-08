# Daily Volatility Regime Report

Date: 2026-06-26T14:01:41.905912+00:00

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v3_1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.08 |
| Mid-Vol Chop | 0.34 |
| Vol Expansion Transition | 0.48 |
| High-Vol Risk-Off | 0.11 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 19.58 | Current implied-vol level |
| VVIX | 94.66 | Vol-of-vol state |
| VVIX/VIX | 4.83 | Convexity stress ratio |
| VVIX/VIX z-score | -1.45 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.148 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `738.00`
- Suggested DTE: `3`
- Notes: HMM transition risk is elevated, so the overwrite duration was shortened.

## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v3.1 Meta-Blend`

Model converged: `True`

Trained and active.

Top HMM state: `VOL_EXPANSION`

Current-state expected duration: `5.25` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.08 |
| MID_VOL_CHOP | 0.34 |
| VOL_EXPANSION | 0.48 |
| HIGH_VOL_STRESS | 0.11 |

### HMM Persistence

- Current state persists 5d: `0.46`
- Current state persists 10d: `0.32`
- Current state persists 21d: `0.24`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.47`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.39`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.29`

Warnings:
- HMM state `HIGH_VOL_STRESS` used less than 5% of the training window.

## Sector Correlation / Market Mode

- avg_pairwise_corr_21d: `0.0000`
- first_eigenvalue_share_21d: `0.0000`

Interpretation:
- Low avg correlation and low first eigenvalue share suggest stable sector independence.


## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | VOL_EXPANSION | 0.89 | 5.25 | 0.07 | MEDIUM_OVERWRITE |
| HMM v2 Core + Sector Corr | VOL_EXPANSION | 0.90 | 9.74 | 0.02 | MEDIUM_OVERWRITE |
| HMM v3 Core + Geometry | VOL_EXPANSION | 0.90 | 8.79 | 0.02 | MEDIUM_OVERWRITE |
| HMM v3.1 Meta-Blend | VOL_EXPANSION | 0.84 | 5.25 | 0.07 | MEDIUM_OVERWRITE |


## Model Confidence

Confidence: 0.84

Uncertainty / entropy: 0.83

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
