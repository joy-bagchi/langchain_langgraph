# Overwrite Candidate Report

- Timestamp: 2026-06-26 11:19:16 UTC
- Underlying: SPY
- Spot: 729.86
- VIX: 20.00
- LEAP contracts: 5
- LEAP delta: 0.80
- Recommendation mode: UNCERTAIN_SELECTIVE
- Daily sigma points: 9.20
- Heuristic target strike: 734.46
- Allowed DTE: [1]
- Recommended action: NO OVERWRITE

## HMM Context

- As of: 2026-06-26T11:19:03.635837+00:00
- Selected regime: VOL_EXPANSION
- crash: 0.00%
- low_vol_trend: 7.59%
- mid_vol_chop: 34.04%
- vol_expansion: 47.72%

## Recommendation

No candidate passed the current decision rules.

## Top Accepted Candidates

No rows.

## Top Rejected Candidates

| strike | dte | mid | distance_sigma | score | reject_reasons |
| --- | --- | --- | --- | --- | --- |
| 730.0 | 5 | 8.29 | 0.02 | 2401.87 | DTE not allowed by policy [1]; Distance sigma below minimum 0.50 |
| 731.0 | 5 | 7.64 | 0.12 | 2138.12 | DTE not allowed by policy [1]; Distance sigma below minimum 0.50 |
| 732.0 | 5 | 6.99 | 0.23 | 1874.37 | DTE not allowed by policy [1]; Distance sigma below minimum 0.50 |
| 730.0 | 4 | 7.44 | 0.02 | 1828.12 | DTE not allowed by policy [1]; Distance sigma below minimum 0.50 |
| 733.0 | 5 | 6.39 | 0.34 | 1644.37 | DTE not allowed by policy [1]; Distance sigma below minimum 0.50 |

## Why Others Failed

- Strike 730.00 / 5DTE rejected because DTE not allowed by policy [1]; Distance sigma below minimum 0.50.
- Strike 731.00 / 5DTE rejected because DTE not allowed by policy [1]; Distance sigma below minimum 0.50.
- Strike 732.00 / 5DTE rejected because DTE not allowed by policy [1]; Distance sigma below minimum 0.50.
- Strike 730.00 / 4DTE rejected because DTE not allowed by policy [1]; Distance sigma below minimum 0.50.
- Strike 733.00 / 5DTE rejected because DTE not allowed by policy [1]; Distance sigma below minimum 0.50.

## Warning

This report is decision support only. It does not execute trades and does not model gamma, vega, skew, early assignment, or liquidity beyond bid/ask spread.
