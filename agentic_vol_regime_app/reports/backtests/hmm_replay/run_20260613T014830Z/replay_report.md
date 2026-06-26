# HMM Replay Backtest Report

## Overall Model Comparison

| model_name                       |   horizon |   accuracy |   brier_vol_expansion |   brier_vix_spike |   avg_lead_quality | notes   |
|:---------------------------------|----------:|-----------:|----------------------:|------------------:|-------------------:|:--------|
| hmm_v3_core_plus_sector_geometry |         1 |   0.3125   |              0.463165 |          0.1508   |           0.479167 |         |
| hmm_v3_core_plus_sector_geometry |         2 |   0.291667 |              0.510124 |          0.160224 |           0.354167 |         |
| hmm_v3_core_plus_sector_geometry |         3 |   0.3125   |              0.447855 |          0.142286 |           0.395833 |         |

## Recent As-of Date Comparison

| As Of      | Model                            | Predicted State          | T+1 Outcome              | T+2 Outcome              | T+3 Outcome              | Score   |
|:-----------|:---------------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:--------|
| 2026-04-27 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | MID_VOL_CHOP             | MID_VOL_CHOP             |         |
| 2026-04-28 | hmm_v3_core_plus_sector_geometry | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |         |
| 2026-04-29 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-04-30 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-01 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-04 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |         |
| 2026-05-05 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-06 | hmm_v3_core_plus_sector_geometry | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-07 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-08 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-11 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |         |
| 2026-05-12 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-13 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |         |
| 2026-05-14 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-15 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-18 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |         |
| 2026-05-19 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-20 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-21 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-22 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-25 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-05-26 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-27 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |         |
| 2026-05-28 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |         |
| 2026-05-29 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |         |
| 2026-06-01 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-02 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-03 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-04 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-05 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |

## Model Disagreement Cases

_No rows_

## HMM v3 Special Section

Track whether geometry features improved false vol-expansion avoidance and mid-vol chop detection.

## Diagnostics

| as_of_date   | model_name                       | converged   |   training_row_count | training_end_date   | warnings   |
|:-------------|:---------------------------------|:------------|---------------------:|:--------------------|:-----------|
| 2026-04-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-01          |            |
| 2026-04-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-02          |            |
| 2026-04-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-03          |            |
| 2026-04-06   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-06          |            |
| 2026-04-07   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-07          |            |
| 2026-04-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-08          |            |
| 2026-04-09   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-09          |            |
| 2026-04-10   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-10          |            |
| 2026-04-13   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-13          |            |
| 2026-04-14   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-14          |            |
| 2026-04-15   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-15          |            |
| 2026-04-16   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-16          |            |
| 2026-04-17   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-17          |            |
| 2026-04-20   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-20          |            |
| 2026-04-21   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-21          |            |
| 2026-04-22   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-22          |            |
| 2026-04-23   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-23          |            |
| 2026-04-24   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-24          |            |
| 2026-04-27   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-27          |            |
| 2026-04-28   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-28          |            |
| 2026-04-29   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-29          |            |
| 2026-04-30   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-30          |            |
| 2026-05-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-01          |            |
| 2026-05-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-04          |            |
| 2026-05-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-05          |            |
| 2026-05-06   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-06          |            |
| 2026-05-07   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-07          |            |
| 2026-05-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-08          |            |
| 2026-05-11   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-11          |            |
| 2026-05-12   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-12          |            |
| 2026-05-13   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-13          |            |
| 2026-05-14   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-14          |            |
| 2026-05-15   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-15          |            |
| 2026-05-18   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-18          |            |
| 2026-05-19   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-19          |            |
| 2026-05-20   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-20          |            |
| 2026-05-21   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-21          |            |
| 2026-05-22   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-22          |            |
| 2026-05-25   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-25          |            |
| 2026-05-26   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-26          |            |
| 2026-05-27   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-27          |            |
| 2026-05-28   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-28          |            |
| 2026-05-29   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-29          |            |
| 2026-06-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-01          |            |
| 2026-06-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-02          |            |
| 2026-06-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-03          |            |
| 2026-06-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-04          |            |
| 2026-06-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-05          |            |
