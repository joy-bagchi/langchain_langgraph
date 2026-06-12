# Daily Volatility Regime Report

Date: 2026-06-12T14:52:17.275494+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Mid-Vol Chop`

Transition risk: `WATCH`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| High-Vol Risk-Off | 0.05 |
| Mid-Vol Chop | 0.31 |
| Panic Convexity Stress | 0.03 |
| Post-Panic Compression | 0.04 |
| Stable Low-Vol Trend | 0.28 |
| Vol Expansion Transition | 0.31 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 19.30 | Current implied-vol level |
| VVIX | 99.15 | Vol-of-vol state |
| VVIX/VIX | 5.14 | Convexity stress ratio |
| VVIX/VIX z-score | -0.57 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.066 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `745.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | MID_VOL_CHOP | 0.69 | MEDIUM_OVERWRITE |
| Linear ML | MID_VOL_CHOP | 0.43 | LIGHT_OVERWRITE |






## Model Confidence

Confidence: 0.69

Uncertainty / entropy: 0.81

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
