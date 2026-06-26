# Daily Volatility Regime Report

Date: 2026-06-23T13:28:42.670399+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.19 |
| Mid-Vol Chop | 0.26 |
| Vol Expansion Transition | 0.48 |
| High-Vol Risk-Off | 0.07 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 20.11 | Current implied-vol level |
| VVIX | 91.79 | Vol-of-vol state |
| VVIX/VIX | 4.56 | Convexity stress ratio |
| VVIX/VIX z-score | -1.42 | Relative convexity stress |
| VIX term structure (VIX3M) | flat | Front/back of curve |
| Realized vol trend | 0.030 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `740.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.




## Model Confidence

Confidence: 0.73

Uncertainty / entropy: 0.87

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
