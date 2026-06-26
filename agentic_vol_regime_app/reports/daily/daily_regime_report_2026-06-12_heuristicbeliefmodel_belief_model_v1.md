# Daily Volatility Regime Report

Date: 2026-06-12T16:36:22.083742+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WATCH`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.35 |
| Mid-Vol Chop | 0.21 |
| Vol Expansion Transition | 0.38 |
| High-Vol Risk-Off | 0.06 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.76 | Current implied-vol level |
| VVIX | 98.07 | Vol-of-vol state |
| VVIX/VIX | 5.23 | Convexity stress ratio |
| VVIX/VIX z-score | -0.29 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.065 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WATCH`

Headline: Transition probability has risen but confirmation remains limited

Drivers:
- realized volatility acceleration
- term structure flattening

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

Confidence: 0.69

Uncertainty / entropy: 0.89

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
