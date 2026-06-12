# Daily Volatility Regime Report

Date: 2026-06-12T23:07:27.640551+00:00

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v2`

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
| VIX | 17.68 | Current implied-vol level |
| VVIX | 94.85 | Vol-of-vol state |
| VVIX/VIX | 5.36 | Convexity stress ratio |
| VVIX/VIX z-score | 0.14 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.131 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `749.00`
- Suggested DTE: `3`
- Notes: HMM transition risk is elevated, so the overwrite duration was shortened.

## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v2 Core + Sector Corr`

Model converged: `True`

Trained and active.

Top HMM state: `VOL_EXPANSION`

Current-state expected duration: `8.12` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.00 |
| MID_VOL_CHOP | 0.00 |
| VOL_EXPANSION | 1.00 |
| HIGH_VOL_STRESS | 0.00 |

### HMM Persistence

- Current state persists 5d: `0.57`
- Current state persists 10d: `0.39`
- Current state persists 21d: `0.27`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.59`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.42`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.30`

Warnings:
- HMM state `HIGH_VOL_STRESS` used less than 5% of the training window.

## Sector Correlation / Market Mode

- avg_pairwise_corr_21d: `0.1471`
- first_eigenvalue_share_21d: `0.4215`

Interpretation:
- Rising avg correlation and rising first eigenvalue share point to increasing vol-expansion risk.


## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | VOL_EXPANSION | 0.90 | 6.59 | 0.10 | MEDIUM_OVERWRITE |
| HMM v2 Core + Sector Corr | VOL_EXPANSION | 0.90 | 8.12 | 0.03 | MEDIUM_OVERWRITE |
| HMM v3 Core + Geometry | VOL_EXPANSION | 0.90 | 8.81 | 0.02 | MEDIUM_OVERWRITE |


## Model Confidence

Confidence: 0.90

Uncertainty / entropy: 0.01

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
