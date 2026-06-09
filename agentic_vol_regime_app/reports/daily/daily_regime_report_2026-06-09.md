# Daily Volatility Regime Report

Date: 2026-06-09T02:48:05.366806+00:00

## Summary

Current regime belief favors: `Stable Low-Vol Trend`

Transition risk: `WATCH`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.54 |
| Mid-Vol Chop | 0.13 |
| Vol Expansion Transition | 0.24 |
| High-Vol Risk-Off | 0.04 |
| Panic Convexity Stress | 0.02 |
| Post-Panic Compression | 0.03 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.92 | Current implied-vol level |
| VVIX | 93.08 | Vol-of-vol state |
| VVIX/VIX | 4.92 | Convexity stress ratio |
| VVIX/VIX z-score | -1.85 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.079 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WATCH`

Headline: Transition probability has risen but confirmation remains limited

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
- Suggested call overwrite strike: `747.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.

## Model Confidence

Confidence: 0.86

Uncertainty / entropy: 0.70

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
