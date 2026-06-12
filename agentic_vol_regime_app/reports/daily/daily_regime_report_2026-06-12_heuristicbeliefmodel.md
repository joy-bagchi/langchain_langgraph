# Daily Volatility Regime Report

Date: 2026-06-12T15:31:08.626029+00:00

Report model: `HeuristicBeliefModel`
Report model version: `belief_model_v1`

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `NONE`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.32 |
| Mid-Vol Chop | 0.19 |
| Vol Expansion Transition | 0.35 |
| High-Vol Risk-Off | 0.05 |
| Panic Convexity Stress | 0.04 |
| Post-Panic Compression | 0.04 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 18.56 | Current implied-vol level |
| VVIX | 97.20 | Vol-of-vol state |
| VVIX/VIX | 5.24 | Convexity stress ratio |
| VVIX/VIX z-score | -0.26 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.066 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

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
- Suggested call overwrite strike: `751.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.


## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
| Heuristic | VOL_EXPANSION_TRANSITION | 0.71 | MEDIUM_OVERWRITE |
| Linear ML | STABLE_LOW_VOL_TREND | 0.48 | NO_OVERWRITE |






## Model Confidence

Confidence: 0.71

Uncertainty / entropy: 0.82

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
