# Daily Volatility Regime Report

Date: 2026-07-09T04:46:24.414966+00:00

Report model: `HMMBeliefAgent`
Report model version: `hmm_gaussian_v3_1`

## Summary

Current regime belief favors: `Mid-Vol Chop`

Transition risk: `NONE`

Recommended posture: `LIGHT_OVERWRITE`

## Belief State

| Regime | Probability |
|---|---:|
| Stable Low-Vol Trend | 0.22 |
| Mid-Vol Chop | 0.62 |
| Vol Expansion Transition | 0.13 |
| High-Vol Risk-Off | 0.03 |

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | 16.90 | Current implied-vol level |
| VVIX | 90.97 | Vol-of-vol state |
| VVIX/VIX | 5.38 | Convexity stress ratio |
| VVIX/VIX z-score | 0.24 | Relative convexity stress |
| VIX term structure (VIX3M) | contango | Front/back of curve |
| Realized vol trend | 0.052 | Short-vs-medium realized vol |

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
- Suggested call overwrite strike: `751.00`
- Suggested DTE: `5`
- Notes: HMM expected regime duration suggests approximately 5 DTE.

## HMM Regime Persistence

Training status: `trained`

Variant: `HMM v3.1 Meta-Blend`

Model converged: `True`

Trained and active.

Top HMM state: `MID_VOL_CHOP`

Current-state expected duration: `4.82` days

### HMM State Probabilities

| State | Probability |
|---|---:|
| LOW_VOL_TREND | 0.22 |
| MID_VOL_CHOP | 0.62 |
| VOL_EXPANSION | 0.13 |
| HIGH_VOL_STRESS | 0.03 |

### HMM Persistence

- Current state persists 5d: `0.45`
- Current state persists 10d: `0.36`
- Current state persists 21d: `0.35`
- VOL_EXPANSION or HIGH_VOL within 5d: `0.22`
- VOL_EXPANSION or HIGH_VOL within 10d: `0.29`
- VOL_EXPANSION or HIGH_VOL within 21d: `0.32`

Warnings:
- None

## Sector Correlation / Market Mode

- avg_pairwise_corr_21d: `0.0000`
- first_eigenvalue_share_21d: `0.0000`

Interpretation:
- Low avg correlation and low first eigenvalue share suggest stable sector independence.


## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
| HMM v1 Core | MID_VOL_CHOP | 0.90 | 4.82 | 0.04 | LIGHT_OVERWRITE |
| HMM v2 Core + Sector Corr | MID_VOL_CHOP | 0.90 | 5.37 | 0.01 | LIGHT_OVERWRITE |
| HMM v3 Core + Geometry | VOL_EXPANSION | 0.90 | 9.21 | 0.02 | MEDIUM_OVERWRITE |
| HMM v3.1 Meta-Blend | MID_VOL_CHOP | 0.85 | 4.82 | 0.04 | LIGHT_OVERWRITE |


## Model Confidence

Confidence: 0.85

Uncertainty / entropy: 0.72

## Critic Review

Verdict: `ALLOW`

Findings:
- Deterministic checks found the daily report candidate internally consistent.

## Required Human Review

Required: No

Review decision: Not required
