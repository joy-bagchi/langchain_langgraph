# Daily Volatility Regime Report

Date: 2026-06-10T05:09:17.389454+00:00

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.27 |
| Mid-Vol Chop | 0.16 |
| Vol Expansion Transition | 0.44 |
| High-Vol Risk-Off | 0.06 |
| Panic Convexity Stress | 0.03 |
| Post-Panic Compression | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 19.87 | Current implied-vol level |
| VVIX | 96.52 | Vol-of-vol state |
| VVIX/VIX | 4.86 | Convexity stress ratio |
| VVIX/VIX z-score | -1.79 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.068 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WARNING`

Headline: Volatility expansion risk is rising with confirming signals

Drivers:
- realized volatility acceleration
- term structure flattening
- IV over realized-vol spread

## Policy Recommendation

Recommended action: `MEDIUM_OVERWRITE`

Rationale:
- Transition risk is elevated and argues for additional premium capture.

Risk notes:
- None

Overwrite implementation:
- Suggested call overwrite strike: `741.00`
- Suggested DTE: `3`
- Notes: HMM transition risk is elevated, so the overwrite duration was shortened.


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | VOL_EXPANSION_TRANSITION | 0.77 | MEDIUM_OVERWRITE |
| Linear ML | MID_VOL_CHOP | 0.43 | MEDIUM_OVERWRITE |
| HMM | HIGH_VOL | 0.90 | AGGRESSIVE_OVERWRITE |
| Ensemble (disabled) | Disabled | 0.00 | Disabled |


## HMM Regime Persistence

Training status: `trained`

Trained and active.

Top HMM state: `HIGH_VOL`

Current-state expected duration: `4.49` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| STABLE | 0.00 |
| EXPANDING_VOL | 0.00 |
| HIGH_VOL | 1.00 |

### Emission vs Persistence

Emission-only top state: `HIGH_VOL`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
| STABLE | 0.00 | 0.00 |
| EXPANDING_VOL | 0.00 | 0.00 |
| HIGH_VOL | 1.00 | -0.00 |

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|
| HIGH_VOL | 23.96 | 14.28 | 0.03 | 0.47 | 0.50 | 5.00 |
| STABLE | 15.97 | 10.19 | 0.00 | 3.16 | 0.62 | 5.96 |
| EXPANDING_VOL | 18.37 | 14.89 | 0.01 | 2.31 | 0.58 | 5.50 |

### Interpretation Notes

- Current features themselves fit `HIGH_VOL` best; emission-only and path-aware posteriors agree.
- `HIGH_VOL` posterior is 1.00 vs emission-only 1.00; persistence lift is -0.00.
- Mapped state summary: VIX 23.96, RV21 14.28, drawdown 0.03, slope 0.47.
- Forward transition risk is elevated: 5d expansion/high-vol probability 0.82, high-vol stress probability 0.42.

### HMM Persistence

- Current state persists 5d: `0.43`
- Current state persists 10d: `0.30`
- Current state persists 21d: `0.22`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.82`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.64`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.51`

### HMM Transition Matrix

| 0.78 | 0.00 | 0.22 |
| 0.02 | 0.92 | 0.06 |
| 0.14 | 0.15 | 0.71 |

Warnings:
- None


## Model Confidence

Confidence: 0.77

Uncertainty / entropy: 0.78

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
