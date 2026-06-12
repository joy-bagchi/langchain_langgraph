# Daily Volatility Regime Report

Date: 2026-05-29T20:00:00Z

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.82 |
| Mid-Vol Chop | 0.03 |
| Vol Expansion Transition | 0.09 |
| High-Vol Risk-Off | 0.03 |
| Panic Convexity Stress | 0.01 |
| Post-Panic Compression | 0.02 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.40 | Current implied-vol level |
| VVIX | 101.00 | Vol-of-vol state |
| VVIX/VIX | 5.49 | Convexity stress ratio |
| VVIX/VIX z-score | -1.63 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | -0.006 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- term structure flattening
- IV over realized-vol spread

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

Overwrite implementation:
- None


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | STABLE_LOW_VOL_TREND | 1.00 | NO_OVERWRITE |
| Linear ML | STABLE_LOW_VOL_TREND | 0.52 | NO_OVERWRITE |
| HMMV1 | Not trained enough | 0.00 | Fallback to heuristic |
| Ensemble (disabled) | Disabled | 0.00 | Disabled |




## HMM Regime Persistence

Training status: `not_trained_enough`

Variant: `HMM v1 Core`

Model converged: `False`

Not trained enough for this run. HMM-specific regime inference is unavailable.

Top HMM state: `NOT_TRAINED_ENOUGH`

Current-state expected duration: `0.00` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| STABLE | 0.00 |
| EXPANDING_VOL | 0.00 |
| HIGH_VOL | 0.00 |

### Emission vs Persistence

Emission-only top state: `NOT_TRAINED_ENOUGH`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
| STABLE | 0.00 | 0.00 |
| EXPANDING_VOL | 0.00 | 0.00 |
| HIGH_VOL | 0.00 | 0.00 |

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|


### Interpretation Notes

- Insufficient aligned history for HMM training or inference.

### HMM Persistence

- Current state persists 5d: `0.00`
- Current state persists 10d: `0.00`
- Current state persists 21d: `0.00`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.00`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.00`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.00`

### HMM Transition Matrix

| 0.00 | 0.00 | 0.00 |
| 0.00 | 0.00 | 0.00 |
| 0.00 | 0.00 | 0.00 |

### State Usage Counts

- STABLE: 0
- EXPANDING_VOL: 0
- HIGH_VOL: 0

### Sector Metrics

- None

Warnings:
- Insufficient aligned history for HMM training or inference.


## Model Confidence

Confidence: 1.00

Uncertainty / entropy: 0.40

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
