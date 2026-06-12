# Daily Volatility Regime Report

Date: 2026-06-12T15:35:34.720832+00:00

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
| Panic Convexity Stress | 0.00 |
| Post-Panic Compression | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.70 | Current implied-vol level |
| VVIX | 97.43 | Vol-of-vol state |
| VVIX/VIX | 5.21 | Convexity stress ratio |
| VVIX/VIX z-score | -0.39 | Relative convexity stress |
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


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | VOL_EXPANSION_TRANSITION | 0.71 | MEDIUM_OVERWRITE |
| Linear ML | STABLE_LOW_VOL_TREND | 0.52 | MEDIUM_OVERWRITE |
| HMMV1 | EXPANDING_VOL | 0.90 | MEDIUM_OVERWRITE |
| Ensemble (disabled) | Disabled | 0.00 | Disabled |



## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | EXPANDING_VOL | 0.90 | 6.99 | 0.11 | MEDIUM_OVERWRITE |
| HMM v1 Core | EXPANDING_VOL | 0.90 | 6.99 | 0.11 | MEDIUM_OVERWRITE |




## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v1 Core`

Model converged: `True`

Trained and active.

Top HMM state: `EXPANDING_VOL`

Current-state expected duration: `6.99` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| STABLE | 0.00 |
| EXPANDING_VOL | 1.00 |
| HIGH_VOL | 0.00 |

### Emission vs Persistence

Emission-only top state: `EXPANDING_VOL`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
| STABLE | 0.00 | 0.00 |
| EXPANDING_VOL | 0.99 | 0.01 |
| HIGH_VOL | 0.01 | -0.01 |

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|
| STABLE | 14.98 | 11.65 | 0.00 | 2.59 | 0.61 | 6.07 |
| EXPANDING_VOL | 18.16 | 13.01 | 0.02 | 1.36 | 0.51 | 5.58 |
| HIGH_VOL | 27.00 | 26.70 | 0.05 | -0.81 | 0.47 | 4.54 |

### Interpretation Notes

- Current features themselves fit `EXPANDING_VOL` best; emission-only and path-aware posteriors agree.
- `EXPANDING_VOL` posterior is 1.00 vs emission-only 0.99; persistence lift is +0.01.
- Mapped state summary: VIX 18.16, RV21 13.01, drawdown 0.02, slope 1.36.
- Forward transition risk is elevated: 5d expansion/high-vol probability 0.63, high-vol stress probability 0.09.

### HMM Persistence

- Current state persists 5d: `0.54`
- Current state persists 10d: `0.41`
- Current state persists 21d: `0.36`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.63`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.52`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.47`

### HMM Transition Matrix

| 0.93 | 0.07 | 0.00 |
| 0.11 | 0.86 | 0.03 |
| 0.00 | 0.12 | 0.88 |

### State Usage Counts

- STABLE: 403
- EXPANDING_VOL: 254
- HIGH_VOL: 73

### Sector Metrics

- None

Warnings:
- None


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
