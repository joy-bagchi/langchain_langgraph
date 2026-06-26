# Overwrite Candidate Report

- Timestamp: 2026-06-26 02:39:24 UTC
- Underlying: SPY
- Spot: 740.25
- VIX: 16.80
- LEAP contracts: 5
- LEAP delta: 0.80
- Recommendation mode: NO_HMM_CONTEXT
- Daily sigma points: 7.83
- Heuristic target strike: 744.17

## Recommendation

Best candidate: SPY 745C 2DTE at mid 1.63. Distance is 0.61 sigma. Premium meets threshold. Portfolio score is -810.42. Recommended only as no hmm context.

## Top Accepted Candidates

| strike | dte | mid | distance_sigma | premium_total | score |
| --- | --- | --- | --- | --- | --- |
| 745 | 2 | 1.63 | 0.61 | 815.00 | -810.42 |
| 743 | 1 | 1.52 | 0.35 | 760.00 | -1234.67 |

## Top Rejected Candidates

| strike | dte | mid | distance_sigma | score | reject_reasons |
| --- | --- | --- | --- | --- | --- |
| 747 | 2 | 1.07 | 0.86 | -838.42 | Premium below minimum 1.40 |
| 744 | 1 | 1.24 | 0.48 | -1248.67 | Premium below minimum 1.40 |
| 742 | 1 | 1.39 | 0.22 | -1497.42 | Premium below minimum 1.40; Distance sigma below minimum 0.35 |

## Best Candidate Scenario PnL

| Scenario Sigma | Scenario Spot | LEAP PnL | Short Call PnL | Total PnL | LEAP Only PnL | Overwrite Drag |
| --- | --- | --- | --- | --- | --- | --- |
| -1.0 | 732.42 | -3,133.63 | 815.00 | -2,318.63 | -3,133.63 | -815.00 |
| -0.5 | 736.33 | -1,566.81 | 815.00 | -751.81 | -1,566.81 | -815.00 |
| 0.0 | 740.25 | 0.00 | 815.00 | 815.00 | 0.00 | -815.00 |
| 0.5 | 744.17 | 1,566.81 | 815.00 | 2,381.81 | 1,566.81 | -815.00 |
| 1.0 | 748.08 | 3,133.63 | -727.03 | 2,406.59 | 3,133.63 | 727.03 |
| 1.5 | 752.00 | 4,700.44 | -2,685.55 | 2,014.89 | 4,700.44 | 2,685.55 |
| 2.0 | 755.92 | 6,267.26 | -4,644.07 | 1,623.19 | 6,267.26 | 4,644.07 |

## Warning

This report is decision support only. It does not execute trades and does not model gamma, vega, skew, early assignment, or liquidity beyond bid/ask spread.
