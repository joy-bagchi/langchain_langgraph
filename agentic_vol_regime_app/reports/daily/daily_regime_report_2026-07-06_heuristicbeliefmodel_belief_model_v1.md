# Daily Volatility Regime Report

Date: 2026-07-06T17:28:50.092234+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.68 |
| Mid-Vol Chop | 0.12 |
| Vol Expansion Transition | 0.15 |
| High-Vol Risk-Off | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 15.86 | Current implied-vol level |
| VVIX | 88.31 | Vol-of-vol state |
| VVIX/VIX | 5.57 | Convexity stress ratio |
| VVIX/VIX z-score | 1.34 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | -0.059 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- VVIX/VIX z-score

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

Overwrite implementation:
- None




## Model Confidence

Confidence: 0.93

Uncertainty / entropy: 0.68

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
