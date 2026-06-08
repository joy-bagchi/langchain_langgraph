# Daily Volatility Regime Report

Date: 2026-06-05T13:51:12.823014+00:00

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.72 |
| Mid-Vol Chop | 0.07 |
| Vol Expansion Transition | 0.08 |
| High-Vol Risk-Off | 0.06 |
| Panic Convexity Stress | 0.03 |
| Post-Panic Compression | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 16.45 | Current implied-vol level |
| VVIX | 88.58 | Vol-of-vol state |
| VVIX/VIX | 5.38 | Convexity stress ratio |
| VVIX/VIX z-score | -0.36 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.012 | Short-vs-medium realized vol |

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
