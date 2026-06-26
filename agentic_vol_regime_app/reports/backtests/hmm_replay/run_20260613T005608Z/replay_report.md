# HMM Replay Backtest Report

## Overall Model Comparison

| model_name                       |   horizon |   accuracy |   brier_vol_expansion |   brier_vix_spike |   avg_lead_quality | notes   |
|:---------------------------------|----------:|-----------:|----------------------:|------------------:|-------------------:|:--------|
| hmm_v3_core_plus_sector_geometry |         1 |   0.5      |              0.299686 |          0.413212 |           0.666667 |         |
| hmm_v3_core_plus_sector_geometry |         2 |   0.166667 |              0.644191 |          0.381698 |           0.166667 |         |
| hmm_v3_core_plus_sector_geometry |         3 |   0.333333 |              0.512029 |          0.539287 |           0.333333 |         |

## Recent As-of Date Comparison

| As Of      | Model                            | Predicted State          | T+1 Outcome              | T+2 Outcome              | T+3 Outcome              | Score   |
|:-----------|:---------------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:--------|
| 2026-06-01 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-02 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-03 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-04 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-05 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |
| 2026-06-08 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |

## Model Disagreement Cases

_No rows_

## HMM v3 Special Section

Track whether geometry features improved false vol-expansion avoidance and mid-vol chop detection.

## Diagnostics

| as_of_date   | model_name                       | converged   |   training_row_count | training_end_date   | warnings   |
|:-------------|:---------------------------------|:------------|---------------------:|:--------------------|:-----------|
| 2026-06-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-01          |            |
| 2026-06-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-02          |            |
| 2026-06-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-03          |            |
| 2026-06-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-04          |            |
| 2026-06-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-05          |            |
| 2026-06-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-08          |            |
