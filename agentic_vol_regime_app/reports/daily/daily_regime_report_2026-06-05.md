# Daily Volatility Regime Report

Date: 2026-06-05T03:45:20.718728+00:00

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `NONE`

Recommended posture: `NO_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.88 |
| Mid-Vol Chop | 0.04 |
| Vol Expansion Transition | 0.03 |
| High-Vol Risk-Off | 0.02 |
| Panic Convexity Stress | 0.01 |
| Post-Panic Compression | 0.02 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 15.40 | Current implied-vol level |
| VVIX | 85.50 | Vol-of-vol state |
| VVIX/VIX | 5.55 | Convexity stress ratio |
| VVIX/VIX z-score | 0.57 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.004 | Short-vs-medium realized vol |

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

Uncertainty / entropy: 0.30

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
