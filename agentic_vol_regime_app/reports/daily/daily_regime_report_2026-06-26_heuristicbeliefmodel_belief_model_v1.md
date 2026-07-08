# Daily Volatility Regime Report

Date: 2026-06-26T10:52:22.088437+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Mid-Vol Chop`

Transition risk: `WATCH`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.20 |
| Mid-Vol Chop | 0.50 |
| Vol Expansion Transition | 0.22 |
| High-Vol Risk-Off | 0.07 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 20.20 | Current implied-vol level |
| VVIX | 91.76 | Vol-of-vol state |
| VVIX/VIX | 4.54 | Convexity stress ratio |
| VVIX/VIX z-score | -1.31 | Relative convexity stress |
| VIX term structure (VIX3M) | flat | Front/back of curve |
| Realized vol trend | -0.034 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WATCH`

Headline: Transition probability has risen but confirmation remains limited

Drivers:
- term structure flattening

## Policy Recommendation

Recommended action: `MEDIUM_OVERWRITE`

Rationale:
- Transition risk is elevated and argues for additional premium capture.

Risk notes:
- None

Overwrite implementation:
- Suggested call overwrite strike: `736.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.




## Model Confidence

Confidence: 0.75

Uncertainty / entropy: 0.87

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
