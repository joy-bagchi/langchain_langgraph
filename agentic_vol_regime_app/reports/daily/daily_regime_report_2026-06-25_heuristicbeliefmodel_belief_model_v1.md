# Daily Volatility Regime Report

Date: 2026-06-25T13:41:16.112531+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.70 |
| Mid-Vol Chop | 0.13 |
| Vol Expansion Transition | 0.13 |
| High-Vol Risk-Off | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.25 | Current implied-vol level |
| VVIX | 92.46 | Vol-of-vol state |
| VVIX/VIX | 5.07 | Convexity stress ratio |
| VVIX/VIX z-score | -0.72 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | -0.027 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- term structure flattening

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

Overwrite implementation:
- None




## Model Confidence

Confidence: 0.95

Uncertainty / entropy: 0.66

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
