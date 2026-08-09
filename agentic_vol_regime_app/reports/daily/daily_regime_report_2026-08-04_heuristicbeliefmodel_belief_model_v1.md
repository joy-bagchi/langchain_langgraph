# Daily Volatility Regime Report

Date: 2026-08-04T10:23:07.201974+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.71 |
| Mid-Vol Chop | 0.13 |
| Vol Expansion Transition | 0.12 |
| High-Vol Risk-Off | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 15.70 | Current implied-vol level |
| VVIX | 90.93 | Vol-of-vol state |
| VVIX/VIX | 5.79 | Convexity stress ratio |
| VVIX/VIX z-score | 0.52 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.070 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- realized volatility acceleration

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

Overwrite implementation:
- None




## Model Confidence

Confidence: 0.96

Uncertainty / entropy: 0.65

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
