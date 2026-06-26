# Daily Volatility Regime Report

Date: 2026-06-06T20:52:56.997298+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.14 |
| Mid-Vol Chop | 0.19 |
| Vol Expansion Transition | 0.52 |
| High-Vol Risk-Off | 0.07 |
| Panic Convexity Stress | 0.03 |
| Post-Panic Compression | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 21.51 | Current implied-vol level |
| VVIX | 99.56 | Vol-of-vol state |
| VVIX/VIX | 4.63 | Convexity stress ratio |
| VVIX/VIX z-score | -3.43 | Relative convexity stress |
| VIX term structure (VIX3M) | flat | Front/back of curve |
| Realized vol trend | 0.072 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `764.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.

## Model Confidence

Confidence: 0.81

Uncertainty / entropy: 0.77

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
