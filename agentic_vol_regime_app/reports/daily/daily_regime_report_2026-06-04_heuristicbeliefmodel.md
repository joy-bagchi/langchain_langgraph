# Daily Volatility Regime Report

Date: 2026-06-04T23:52:17.927026+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.87 |
| Mid-Vol Chop | 0.04 |
| Vol Expansion Transition | 0.04 |
| High-Vol Risk-Off | 0.03 |
| Panic Convexity Stress | 0.01 |
| Post-Panic Compression | 0.02 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.40 | Current implied-vol level |
| VVIX | 101.00 | Vol-of-vol state |
| VVIX/VIX | 5.49 | Convexity stress ratio |
| VVIX/VIX z-score | 0.21 | Relative convexity stress |
| VIX term structure | contango | Front/back of curve |
| Realized vol trend | 0.005 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- IV over realized-vol spread

## Policy Recommendation

Recommended action: `NO_OVERWRITE`

Rationale:
- Stable low-volatility trend remains the dominant regime.

Risk notes:
- Tight overwrites may truncate upside more than they help.

## Model Confidence

Confidence: 1.00

Uncertainty / entropy: 0.34

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
