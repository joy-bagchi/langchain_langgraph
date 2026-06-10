# Daily Volatility Regime Report

Date: 2026-06-10T03:49:06.031426+00:00

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.00 |
| Mid-Vol Chop | 0.00 |
| Vol Expansion Transition | 0.98 |
| High-Vol Risk-Off | 0.02 |
| Panic Convexity Stress | 0.00 |
| Post-Panic Compression | 0.00 |

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
| HMM | VOL_EXPANSION | 0.89 | MEDIUM_OVERWRITE |
| Ensemble (disabled) | Disabled | 0.00 | Disabled |


## HMM Regime Persistence

Training status: `trained`

Trained and active.

Top HMM state: `VOL_EXPANSION`

Current-state expected duration: `4.12` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.00 |
| MID_VOL_CHOP | 0.00 |
| VOL_EXPANSION | 0.98 |
| HIGH_VOL_STRESS | 0.02 |

### Emission vs Persistence

Emission-only top state: `VOL_EXPANSION`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
| LOW_VOL_TREND | 0.00 | 0.00 |
| MID_VOL_CHOP | 0.00 | 0.00 |
| VOL_EXPANSION | 0.86 | 0.12 |
| HIGH_VOL_STRESS | 0.14 | -0.12 |

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|
| VOL_EXPANSION | 18.99 | 14.93 | 0.01 | 2.03 | 0.55 | 5.47 |
| HIGH_VOL_STRESS | 24.91 | 14.27 | 0.04 | 0.26 | 0.50 | 4.87 |
| LOW_VOL_TREND | 16.40 | 8.85 | 0.00 | 2.74 | 0.57 | 6.05 |
| MID_VOL_CHOP | 16.00 | 11.27 | 0.00 | 3.26 | 0.64 | 5.87 |

### Interpretation Notes

- Current features themselves fit `VOL_EXPANSION` best; emission-only and path-aware posteriors agree.
- `VOL_EXPANSION` posterior is 0.98 vs emission-only 0.86; persistence lift is +0.12.
- Mapped state summary: VIX 18.99, RV21 14.93, drawdown 0.01, slope 2.03.
- Forward transition risk is elevated: 5d expansion/high-vol probability 0.55, high-vol stress probability 0.18.

### HMM Persistence

- Current state persists 5d: `0.37`
- Current state persists 10d: `0.28`
- Current state persists 21d: `0.26`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.55`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.46`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.42`

### HMM Transition Matrix

| 0.76 | 0.07 | 0.05 | 0.11 |
| 0.18 | 0.82 | 0.00 | 0.00 |
| 0.10 | 0.03 | 0.79 | 0.08 |
| 0.04 | 0.01 | 0.05 | 0.89 |

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
