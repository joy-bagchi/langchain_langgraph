# Daily Volatility Regime Report

Date: 2026-06-05T03:23:45.318898+00:00

## Summary

Current regime belief favors: `Vol Expansion Transition`

Transition risk: `WARNING`

Recommended posture: `MEDIUM_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.23 |
| Mid-Vol Chop | 0.13 |
| Vol Expansion Transition | 0.46 |
| High-Vol Risk-Off | 0.13 |
| Panic Convexity Stress | 0.02 |
| Post-Panic Compression | 0.03 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 21.60 | Current implied-vol level |
| VVIX | 129.60 | Vol-of-vol state |
| VVIX/VIX | 6.00 | Convexity stress ratio |
| VVIX/VIX z-score | 2.53 | Relative convexity stress |
| VIX term structure (VIX3M) | backwardation | Front/back of curve |
| Realized vol trend | 0.004 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `WARNING`

Headline: Volatility expansion risk is rising with confirming signals

Drivers:
- VVIX/VIX z-score
- term structure flattening
- IV over realized-vol spread
- term structure entered backwardation

## Policy Recommendation

Recommended action: `MEDIUM_OVERWRITE`

Rationale:
- Transition risk is elevated and argues for additional premium capture.

Risk notes:
- None

Overwrite implementation:
- Suggested call overwrite strike: `760.00`
- Suggested DTE: `1`
- Notes: Medium overwrite moves the call further out to balance premium and upside room.

## Model Confidence

Confidence: 0.78

Uncertainty / entropy: 0.78

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
