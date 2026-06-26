# Daily Volatility Regime Report

Date: 2026-06-12T23:17:28.201832+00:00

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v1`

## Summary

Current regime belief favors: `Mid-Vol Chop`

Transition risk: `NONE`

Recommended posture: `LIGHT_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.00 |
| Mid-Vol Chop | 1.00 |
| Vol Expansion Transition | 0.00 |
| High-Vol Risk-Off | 0.00 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 17.68 | Current implied-vol level |
| VVIX | 94.85 | Vol-of-vol state |
| VVIX/VIX | 5.36 | Convexity stress ratio |
| VVIX/VIX z-score | 0.15 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.065 | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `NONE`

Headline: No elevated predictive volatility alert

Drivers:
- realized volatility acceleration

## Policy Recommendation

Recommended action: `LIGHT_OVERWRITE`

Rationale:
- Mid-volatility chop favors collecting premium without leaning too aggressive.

Risk notes:
- Stay flexible because chop can resolve in either direction.

Overwrite implementation:
- Suggested call overwrite strike: `747.00`
- Suggested DTE: `5`
- Notes: HMM expected regime duration suggests approximately 5 DTE.

## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v1 Core`

Model converged: `True`

Trained and active.

Top HMM state: `MID_VOL_CHOP`

Current-state expected duration: `5.05` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.00 |
| MID_VOL_CHOP | 1.00 |
| VOL_EXPANSION | 0.00 |
| HIGH_VOL_STRESS | 0.00 |

### HMM Persistence

- Current state persists 5d: `0.46`
- Current state persists 10d: `0.38`
- Current state persists 21d: `0.36`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.24`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.30`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.33`

Warnings:
- Missing sector history for: XLK, XLF, XLE, XLY, XLP, XLI, XLB, XLV, XLU, XLRE
- Insufficient fully populated feature rows for HMM training.
- Fell back to HMM v1 because HMM v3 Core + Geometry sector/geometry data was incomplete.


## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | MID_VOL_CHOP | 0.90 | 5.05 | 0.05 | LIGHT_OVERWRITE |
| HMM v2 Core + Sector Corr | MID_VOL_CHOP | 0.90 | 5.05 | 0.05 | LIGHT_OVERWRITE |
| HMM v3 Core + Geometry | MID_VOL_CHOP | 0.90 | 5.05 | 0.05 | LIGHT_OVERWRITE |


## Model Confidence

Confidence: 0.90

Uncertainty / entropy: 0.02

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
