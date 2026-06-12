# Daily Volatility Regime Report

Date: 2026-06-11T12:14:34.021699+00:00

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
| Vol Expansion Transition | 0.95 |
| High-Vol Risk-Off | 0.05 |
| Panic Convexity Stress | 0.00 |
| Post-Panic Compression | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 20.72 | Current implied-vol level |
| VVIX | 107.85 | Vol-of-vol state |
| VVIX/VIX | 5.21 | Convexity stress ratio |
| VVIX/VIX z-score | -0.43 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.069 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WARNING`

Headline: Volatility expansion risk is rising with confirming signals

Drivers:
- realized volatility acceleration
- term structure flattening
- SPY drawdown pressure
- IV over realized-vol spread

## Policy Recommendation

Recommended action: `MEDIUM_OVERWRITE`

Rationale:
- Transition risk is elevated and argues for additional premium capture.

Risk notes:
- None

Overwrite implementation:
- Suggested call overwrite strike: `737.00`
- Suggested DTE: `3`
- Notes: HMM transition risk is elevated, so the overwrite duration was shortened.


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | VOL_EXPANSION_TRANSITION | 0.69 | MEDIUM_OVERWRITE |
| Linear ML | VOL_EXPANSION_TRANSITION | 0.46 | MEDIUM_OVERWRITE |
| HMM | EXPANDING_VOL | 0.86 | MEDIUM_OVERWRITE |
| Ensemble (disabled) | Disabled | 0.00 | Disabled |


## HMM Regime Persistence

Training status: `trained`

Trained and active.

Top HMM state: `EXPANDING_VOL`

Current-state expected duration: `6.95` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| STABLE | 0.00 |
| EXPANDING_VOL | 0.95 |
| HIGH_VOL | 0.05 |

### Emission vs Persistence

Emission-only top state: `HIGH_VOL`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
| STABLE | 0.00 | 0.00 |
| EXPANDING_VOL | 0.39 | 0.56 |
| HIGH_VOL | 0.61 | -0.56 |

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|
| STABLE | 14.98 | 11.65 | 0.00 | 2.59 | 0.61 | 6.07 |
| EXPANDING_VOL | 18.13 | 13.00 | 0.02 | 1.36 | 0.51 | 5.58 |
| HIGH_VOL | 26.98 | 26.64 | 0.05 | -0.81 | 0.47 | 4.54 |

### Interpretation Notes

- Current features fit `HIGH_VOL` best, but transition persistence lifts the final HMM call to `EXPANDING_VOL`.
- `EXPANDING_VOL` posterior is 0.95 vs emission-only 0.39; persistence lift is +0.56.
- Mapped state summary: VIX 18.13, RV21 13.00, drawdown 0.02, slope 1.36.
- Forward transition risk is elevated: 5d expansion/high-vol probability 0.64, high-vol stress probability 0.12.

### HMM Persistence

- Current state persists 5d: `0.54`
- Current state persists 10d: `0.41`
- Current state persists 21d: `0.36`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.64`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.53`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.47`

### HMM Transition Matrix

| 0.93 | 0.07 | 0.00 |
| 0.11 | 0.86 | 0.03 |
| 0.00 | 0.12 | 0.88 |

Warnings:
- None


## Model Confidence

Confidence: 0.86

Uncertainty / entropy: 0.12

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
