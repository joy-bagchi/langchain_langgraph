# Daily Volatility Regime Report

Date: 2026-07-09T04:36:44.845699+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.76 |
| Mid-Vol Chop | 0.14 |
| Vol Expansion Transition | 0.06 |
| High-Vol Risk-Off | 0.05 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 16.90 | Current implied-vol level |
| VVIX | 90.97 | Vol-of-vol state |
| VVIX/VIX | 5.38 | Convexity stress ratio |
| VVIX/VIX z-score | 0.76 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | -0.069 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- None

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

Overwrite implementation:
- None




## Model Confidence

Confidence: 1.00

Uncertainty / entropy: 0.57

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
