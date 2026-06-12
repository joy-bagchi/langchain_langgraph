# Daily Volatility Regime Report

Date: 2026-06-12T21:43:39.584120+00:00

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
| Mid-Vol Chop | 0.01 |
| Vol Expansion Transition | 0.99 |
| High-Vol Risk-Off | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 17.68 | Current implied-vol level |
| VVIX | 94.85 | Vol-of-vol state |
| VVIX/VIX | 5.36 | Convexity stress ratio |
| VVIX/VIX z-score | 0.13 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.137 | Short-vs-medium realized vol |

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

Variant: `HMM v1 Core`

Model converged: `True`

Trained and active.

Top HMM state: `VOL_EXPANSION`

Current-state expected duration: `6.59` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.00 |
| MID_VOL_CHOP | 0.01 |
| VOL_EXPANSION | 0.99 |
| HIGH_VOL_STRESS | 0.00 |

### HMM Persistence

- Current state persists 5d: `0.51`
- Current state persists 10d: `0.37`
- Current state persists 21d: `0.29`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.60`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.46`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.37`

Warnings:
- None


## Model Confidence

Confidence: 0.89

Uncertainty / entropy: 0.05

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
