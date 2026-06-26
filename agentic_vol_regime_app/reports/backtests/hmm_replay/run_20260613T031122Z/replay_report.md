# HMM Replay Backtest Report

## Overall Model Comparison

| model_name                       |   horizon |   accuracy |   adjacent_tolerant_accuracy |   severe_miss_rate |   brier_vol_expansion |   brier_vix_spike |   risk_bucket_accuracy |   false_alarm_rate |   missed_risk_rate |   vix_directional_accuracy |   vvix_directional_accuracy |   rv_directional_accuracy |   combined_vol_directional_accuracy |   avg_lead_quality | notes   |
|:---------------------------------|----------:|-----------:|-----------------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-------------------:|-------------------:|---------------------------:|----------------------------:|--------------------------:|------------------------------------:|-------------------:|:--------|
| heuristic                        |         1 |   0.506849 |                     0.767123 |           0.232877 |              0.296069 |          0.469759 |               0.534247 |           0.465753 |           0        |                   0.465753 |                    0.520548 |                  0.684932 |                            0.557078 |           0.643836 |         |
| heuristic                        |         2 |   0.493151 |                     0.657534 |           0.342466 |              0.288817 |          0.520783 |               0.547945 |           0.452055 |           0        |                   0.465753 |                    0.520548 |                  0.575342 |                            0.520548 |           0.671233 |         |
| heuristic                        |         3 |   0.472222 |                     0.652778 |           0.347222 |              0.260308 |          0.549566 |               0.555556 |           0.444444 |           0        |                   0.5      |                    0.527778 |                  0.569444 |                            0.532407 |           0.708333 |         |
| hmm_v1_core                      |         1 |   0.410959 |                     0.726027 |           0.273973 |              0.42541  |          0.272077 |               0.561644 |           0.164384 |           0.273973 |                   0.60274  |                    0.60274  |                  0.712329 |                            0.639269 |           0.534247 |         |
| hmm_v1_core                      |         2 |   0.356164 |                     0.657534 |           0.342466 |              0.427617 |          0.27029  |               0.520548 |           0.178082 |           0.30137  |                   0.589041 |                    0.60274  |                  0.630137 |                            0.607306 |           0.506849 |         |
| hmm_v1_core                      |         3 |   0.388889 |                     0.694444 |           0.305556 |              0.411763 |          0.223096 |               0.611111 |           0.125    |           0.263889 |                   0.597222 |                    0.583333 |                  0.652778 |                            0.611111 |           0.541667 |         |
| hmm_v2_core_plus_sector_corr     |         1 |   0.424658 |                     0.753425 |           0.246575 |              0.397178 |          0.288939 |               0.547945 |           0.178082 |           0.273973 |                   0.616438 |                    0.616438 |                  0.712329 |                            0.648402 |           0.547945 |         |
| hmm_v2_core_plus_sector_corr     |         2 |   0.369863 |                     0.671233 |           0.328767 |              0.384767 |          0.264261 |               0.534247 |           0.178082 |           0.287671 |                   0.60274  |                    0.616438 |                  0.630137 |                            0.616438 |           0.520548 |         |
| hmm_v2_core_plus_sector_corr     |         3 |   0.416667 |                     0.708333 |           0.291667 |              0.345392 |          0.220969 |               0.625    |           0.125    |           0.25     |                   0.611111 |                    0.569444 |                  0.666667 |                            0.615741 |           0.555556 |         |
| hmm_v3_core_plus_sector_geometry |         1 |   0.383562 |                     0.712329 |           0.287671 |              0.406443 |          0.269712 |               0.561644 |           0.164384 |           0.273973 |                   0.575342 |                    0.575342 |                  0.712329 |                            0.621005 |           0.534247 |         |
| hmm_v3_core_plus_sector_geometry |         2 |   0.356164 |                     0.60274  |           0.39726  |              0.411241 |          0.278502 |               0.493151 |           0.191781 |           0.315068 |                   0.561644 |                    0.575342 |                  0.616438 |                            0.584475 |           0.479452 |         |
| hmm_v3_core_plus_sector_geometry |         3 |   0.375    |                     0.652778 |           0.347222 |              0.368801 |          0.215075 |               0.583333 |           0.138889 |           0.277778 |                   0.569444 |                    0.555556 |                  0.638889 |                            0.587963 |           0.513889 |         |

## Prediction Distribution

| model_name                       |   horizon | predicted_state          |   count |   percent |
|:---------------------------------|----------:|:-------------------------|--------:|----------:|
| heuristic                        |         1 | VOL_EXPANSION_TRANSITION |      73 | 1         |
| heuristic                        |         2 | VOL_EXPANSION_TRANSITION |      73 | 1         |
| heuristic                        |         3 | VOL_EXPANSION_TRANSITION |      72 | 1         |
| hmm_v1_core                      |         1 | HIGH_VOL_RISK_OFF        |       5 | 0.0684932 |
| hmm_v1_core                      |         1 | MID_VOL_CHOP             |      13 | 0.178082  |
| hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND     |      29 | 0.39726   |
| hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION |      26 | 0.356164  |
| hmm_v1_core                      |         2 | HIGH_VOL_RISK_OFF        |       5 | 0.0684932 |
| hmm_v1_core                      |         2 | MID_VOL_CHOP             |      13 | 0.178082  |
| hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND     |      29 | 0.39726   |
| hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION |      26 | 0.356164  |
| hmm_v1_core                      |         3 | HIGH_VOL_RISK_OFF        |       5 | 0.0694444 |
| hmm_v1_core                      |         3 | MID_VOL_CHOP             |      13 | 0.180556  |
| hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND     |      29 | 0.402778  |
| hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION |      25 | 0.347222  |
| hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP             |      16 | 0.219178  |
| hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND     |      25 | 0.342466  |
| hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION |      32 | 0.438356  |
| hmm_v2_core_plus_sector_corr     |         2 | MID_VOL_CHOP             |      16 | 0.219178  |
| hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND     |      25 | 0.342466  |
| hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION |      32 | 0.438356  |
| hmm_v2_core_plus_sector_corr     |         3 | MID_VOL_CHOP             |      16 | 0.222222  |
| hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND     |      25 | 0.347222  |
| hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION |      31 | 0.430556  |
| hmm_v3_core_plus_sector_geometry |         1 | MID_VOL_CHOP             |      12 | 0.164384  |
| hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND     |      30 | 0.410959  |
| hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION |      31 | 0.424658  |
| hmm_v3_core_plus_sector_geometry |         2 | MID_VOL_CHOP             |      12 | 0.164384  |
| hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND     |      30 | 0.410959  |
| hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION |      31 | 0.424658  |
| hmm_v3_core_plus_sector_geometry |         3 | MID_VOL_CHOP             |      12 | 0.166667  |
| hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND     |      30 | 0.416667  |
| hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION |      30 | 0.416667  |

## Outcome Distribution

|   horizon | realized_state           |   count |   percent |
|----------:|:-------------------------|--------:|----------:|
|         1 | HIGH_VOL_RISK_OFF        |       2 | 0.0273973 |
|         1 | MID_VOL_CHOP             |      17 | 0.232877  |
|         1 | STABLE_LOW_VOL_TREND     |      17 | 0.232877  |
|         1 | VOL_EXPANSION_TRANSITION |      37 | 0.506849  |
|         2 | HIGH_VOL_RISK_OFF        |       4 | 0.0547945 |
|         2 | MID_VOL_CHOP             |       8 | 0.109589  |
|         2 | STABLE_LOW_VOL_TREND     |      25 | 0.342466  |
|         2 | VOL_EXPANSION_TRANSITION |      36 | 0.493151  |
|         3 | HIGH_VOL_RISK_OFF        |       6 | 0.0833333 |
|         3 | MID_VOL_CHOP             |       7 | 0.0972222 |
|         3 | STABLE_LOW_VOL_TREND     |      25 | 0.347222  |
|         3 | VOL_EXPANSION_TRANSITION |      34 | 0.472222  |

## Confusion Matrix by Horizon

| model_name                       |   horizon | predicted_state          | realized_state           |   count |   row_percent |
|:---------------------------------|----------:|:-------------------------|:-------------------------|--------:|--------------:|
| heuristic                        |         1 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       2 |     0.0273973 |
| heuristic                        |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |      17 |     0.232877  |
| heuristic                        |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |      17 |     0.232877  |
| heuristic                        |         1 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      37 |     0.506849  |
| heuristic                        |         2 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       4 |     0.0547945 |
| heuristic                        |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       8 |     0.109589  |
| heuristic                        |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |      25 |     0.342466  |
| heuristic                        |         2 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      36 |     0.493151  |
| heuristic                        |         3 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       6 |     0.0833333 |
| heuristic                        |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       7 |     0.0972222 |
| heuristic                        |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |      25 |     0.347222  |
| heuristic                        |         3 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      34 |     0.472222  |
| hmm_v1_core                      |         1 | HIGH_VOL_RISK_OFF        | MID_VOL_CHOP             |       1 |     0.2       |
| hmm_v1_core                      |         1 | HIGH_VOL_RISK_OFF        | STABLE_LOW_VOL_TREND     |       1 |     0.2       |
| hmm_v1_core                      |         1 | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |       3 |     0.6       |
| hmm_v1_core                      |         1 | MID_VOL_CHOP             | MID_VOL_CHOP             |       5 |     0.384615  |
| hmm_v1_core                      |         1 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       3 |     0.230769  |
| hmm_v1_core                      |         1 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       5 |     0.384615  |
| hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       1 |     0.0344828 |
| hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       4 |     0.137931  |
| hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |      10 |     0.344828  |
| hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      14 |     0.482759  |
| hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       1 |     0.0384615 |
| hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       7 |     0.269231  |
| hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       3 |     0.115385  |
| hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      15 |     0.576923  |
| hmm_v1_core                      |         2 | HIGH_VOL_RISK_OFF        | STABLE_LOW_VOL_TREND     |       2 |     0.4       |
| hmm_v1_core                      |         2 | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |       3 |     0.6       |
| hmm_v1_core                      |         2 | MID_VOL_CHOP             | MID_VOL_CHOP             |       2 |     0.153846  |
| hmm_v1_core                      |         2 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       5 |     0.384615  |
| hmm_v1_core                      |         2 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       6 |     0.461538  |
| hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       2 |     0.0689655 |
| hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       2 |     0.0689655 |
| hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |      11 |     0.37931   |
| hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      14 |     0.482759  |
| hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       2 |     0.0769231 |
| hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       4 |     0.153846  |
| hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       7 |     0.269231  |
| hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      13 |     0.5       |
| hmm_v1_core                      |         3 | HIGH_VOL_RISK_OFF        | MID_VOL_CHOP             |       1 |     0.2       |
| hmm_v1_core                      |         3 | HIGH_VOL_RISK_OFF        | STABLE_LOW_VOL_TREND     |       1 |     0.2       |
| hmm_v1_core                      |         3 | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |       3 |     0.6       |
| hmm_v1_core                      |         3 | MID_VOL_CHOP             | MID_VOL_CHOP             |       2 |     0.153846  |
| hmm_v1_core                      |         3 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       7 |     0.538462  |
| hmm_v1_core                      |         3 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       4 |     0.307692  |
| hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       2 |     0.0689655 |
| hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       2 |     0.0689655 |
| hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |      12 |     0.413793  |
| hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      13 |     0.448276  |
| hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       4 |     0.16      |
| hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       2 |     0.08      |
| hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       5 |     0.2       |
| hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      14 |     0.56      |
| hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP             | MID_VOL_CHOP             |       5 |     0.3125    |
| hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       5 |     0.3125    |
| hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       6 |     0.375     |
| hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       1 |     0.04      |
| hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       3 |     0.12      |
| hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       8 |     0.32      |
| hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      13 |     0.52      |
| hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       1 |     0.03125   |
| hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       9 |     0.28125   |
| hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       4 |     0.125     |
| hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      18 |     0.5625    |
| hmm_v2_core_plus_sector_corr     |         2 | MID_VOL_CHOP             | MID_VOL_CHOP             |       2 |     0.125     |
| hmm_v2_core_plus_sector_corr     |         2 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       8 |     0.5       |
| hmm_v2_core_plus_sector_corr     |         2 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       6 |     0.375     |
| hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       2 |     0.08      |
| hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       2 |     0.08      |
| hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       8 |     0.32      |
| hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      13 |     0.52      |
| hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       2 |     0.0625    |
| hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       4 |     0.125     |
| hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       9 |     0.28125   |
| hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      17 |     0.53125   |
| hmm_v2_core_plus_sector_corr     |         3 | MID_VOL_CHOP             | MID_VOL_CHOP             |       3 |     0.1875    |
| hmm_v2_core_plus_sector_corr     |         3 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |      10 |     0.625     |
| hmm_v2_core_plus_sector_corr     |         3 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       3 |     0.1875    |
| hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       2 |     0.08      |
| hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       1 |     0.04      |
| hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       9 |     0.36      |
| hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      13 |     0.52      |
| hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       4 |     0.129032  |
| hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       3 |     0.0967742 |
| hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       6 |     0.193548  |
| hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      18 |     0.580645  |
| hmm_v3_core_plus_sector_geometry |         1 | MID_VOL_CHOP             | MID_VOL_CHOP             |       3 |     0.25      |
| hmm_v3_core_plus_sector_geometry |         1 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       6 |     0.5       |
| hmm_v3_core_plus_sector_geometry |         1 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       3 |     0.25      |
| hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       1 |     0.0333333 |
| hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       6 |     0.2       |
| hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       7 |     0.233333  |
| hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      16 |     0.533333  |
| hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       1 |     0.0322581 |
| hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       8 |     0.258065  |
| hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       4 |     0.129032  |
| hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      18 |     0.580645  |
| hmm_v3_core_plus_sector_geometry |         2 | MID_VOL_CHOP             | MID_VOL_CHOP             |       2 |     0.166667  |
| hmm_v3_core_plus_sector_geometry |         2 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       6 |     0.5       |
| hmm_v3_core_plus_sector_geometry |         2 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       4 |     0.333333  |
| hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       2 |     0.0666667 |
| hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       2 |     0.0666667 |
| hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       9 |     0.3       |
| hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      17 |     0.566667  |
| hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       2 |     0.0645161 |
| hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       4 |     0.129032  |
| hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |      10 |     0.322581  |
| hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      15 |     0.483871  |
| hmm_v3_core_plus_sector_geometry |         3 | MID_VOL_CHOP             | MID_VOL_CHOP             |       1 |     0.0833333 |
| hmm_v3_core_plus_sector_geometry |         3 | MID_VOL_CHOP             | STABLE_LOW_VOL_TREND     |       9 |     0.75      |
| hmm_v3_core_plus_sector_geometry |         3 | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |       2 |     0.166667  |
| hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        |       3 |     0.1       |
| hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND     | MID_VOL_CHOP             |       3 |     0.1       |
| hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     |       9 |     0.3       |
| hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION |      15 |     0.5       |
| hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |       3 |     0.1       |
| hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             |       3 |     0.1       |
| hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     |       7 |     0.233333  |
| hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |      17 |     0.566667  |

## Economic Score Summary

| model_name                       |   horizon |   accuracy |   adjacent_tolerant_accuracy |   severe_miss_rate |   brier_vol_expansion |   brier_vix_spike |   risk_bucket_accuracy |   false_alarm_rate |   missed_risk_rate |   vix_directional_accuracy |   vvix_directional_accuracy |   rv_directional_accuracy |   combined_vol_directional_accuracy |   avg_lead_quality | notes   |
|:---------------------------------|----------:|-----------:|-----------------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-------------------:|-------------------:|---------------------------:|----------------------------:|--------------------------:|------------------------------------:|-------------------:|:--------|
| heuristic                        |         1 |   0.506849 |                     0.767123 |           0.232877 |              0.296069 |          0.469759 |               0.534247 |           0.465753 |           0        |                   0.465753 |                    0.520548 |                  0.684932 |                            0.557078 |           0.643836 |         |
| heuristic                        |         2 |   0.493151 |                     0.657534 |           0.342466 |              0.288817 |          0.520783 |               0.547945 |           0.452055 |           0        |                   0.465753 |                    0.520548 |                  0.575342 |                            0.520548 |           0.671233 |         |
| heuristic                        |         3 |   0.472222 |                     0.652778 |           0.347222 |              0.260308 |          0.549566 |               0.555556 |           0.444444 |           0        |                   0.5      |                    0.527778 |                  0.569444 |                            0.532407 |           0.708333 |         |
| hmm_v1_core                      |         1 |   0.410959 |                     0.726027 |           0.273973 |              0.42541  |          0.272077 |               0.561644 |           0.164384 |           0.273973 |                   0.60274  |                    0.60274  |                  0.712329 |                            0.639269 |           0.534247 |         |
| hmm_v1_core                      |         2 |   0.356164 |                     0.657534 |           0.342466 |              0.427617 |          0.27029  |               0.520548 |           0.178082 |           0.30137  |                   0.589041 |                    0.60274  |                  0.630137 |                            0.607306 |           0.506849 |         |
| hmm_v1_core                      |         3 |   0.388889 |                     0.694444 |           0.305556 |              0.411763 |          0.223096 |               0.611111 |           0.125    |           0.263889 |                   0.597222 |                    0.583333 |                  0.652778 |                            0.611111 |           0.541667 |         |
| hmm_v2_core_plus_sector_corr     |         1 |   0.424658 |                     0.753425 |           0.246575 |              0.397178 |          0.288939 |               0.547945 |           0.178082 |           0.273973 |                   0.616438 |                    0.616438 |                  0.712329 |                            0.648402 |           0.547945 |         |
| hmm_v2_core_plus_sector_corr     |         2 |   0.369863 |                     0.671233 |           0.328767 |              0.384767 |          0.264261 |               0.534247 |           0.178082 |           0.287671 |                   0.60274  |                    0.616438 |                  0.630137 |                            0.616438 |           0.520548 |         |
| hmm_v2_core_plus_sector_corr     |         3 |   0.416667 |                     0.708333 |           0.291667 |              0.345392 |          0.220969 |               0.625    |           0.125    |           0.25     |                   0.611111 |                    0.569444 |                  0.666667 |                            0.615741 |           0.555556 |         |
| hmm_v3_core_plus_sector_geometry |         1 |   0.383562 |                     0.712329 |           0.287671 |              0.406443 |          0.269712 |               0.561644 |           0.164384 |           0.273973 |                   0.575342 |                    0.575342 |                  0.712329 |                            0.621005 |           0.534247 |         |
| hmm_v3_core_plus_sector_geometry |         2 |   0.356164 |                     0.60274  |           0.39726  |              0.411241 |          0.278502 |               0.493151 |           0.191781 |           0.315068 |                   0.561644 |                    0.575342 |                  0.616438 |                            0.584475 |           0.479452 |         |
| hmm_v3_core_plus_sector_geometry |         3 |   0.375    |                     0.652778 |           0.347222 |              0.368801 |          0.215075 |               0.583333 |           0.138889 |           0.277778 |                   0.569444 |                    0.555556 |                  0.638889 |                            0.587963 |           0.513889 |         |

## False Alarms

| as_of_date   | model_name                       |   horizon | predicted_state          | realized_state       |   vix_change_pct |   vvix_change_pct |   rv21_change |   spy_return_pct |   vix |   vvix_vix_ratio |   term_structure_slope |   avg_pairwise_corr_21d |   first_eigenvalue_share_21d |   effective_rank_21d |   log_det_corr_21d |
|:-------------|:---------------------------------|----------:|:-------------------------|:---------------------|-----------------:|------------------:|--------------:|-----------------:|------:|-----------------:|-----------------------:|------------------------:|-----------------------------:|---------------------:|-------------------:|
| 2026-03-05   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |       0.00763683 |      -0.000775728 |    -0.245979  |      0.00360469  | 23.57 |          4.92236 |                  -0.03 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-05   | hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |       0.00763683 |      -0.000775728 |    -0.245979  |      0.00360469  | 23.57 |          4.92236 |                  -0.03 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-05   | hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |       0.00763683 |      -0.000775728 |    -0.245979  |      0.00360469  | 23.57 |          4.92236 |                  -0.03 |                0.267487 |                     0.381521 |            nan       |          nan       |
| 2026-03-05   | hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |       0.00763683 |      -0.000775728 |    -0.245979  |      0.00360469  | 23.57 |          4.92236 |                  -0.03 |                0.267487 |                     0.381521 |              5.64917 |           -7.07271 |
| 2026-03-10   | heuristic                        |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.1353     |      -0.127029    |    -1.44425   |      0.00601817  | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.154629   |      -0.107377    |    -4.25183   |      0.00990615  | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | heuristic                        |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.178366   |      -0.127813    |    -4.02828   |     -7.44823e-05 | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | hmm_v1_core                      |         1 | HIGH_VOL_RISK_OFF        | STABLE_LOW_VOL_TREND |      -0.1353     |      -0.127029    |    -1.44425   |      0.00601817  | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | hmm_v1_core                      |         2 | HIGH_VOL_RISK_OFF        | STABLE_LOW_VOL_TREND |      -0.154629   |      -0.107377    |    -4.25183   |      0.00990615  | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | hmm_v1_core                      |         3 | HIGH_VOL_RISK_OFF        | MID_VOL_CHOP         |      -0.178366   |      -0.127813    |    -4.02828   |     -7.44823e-05 | 29.49 |          4.76229 |                  -1.93 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-10   | hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.1353     |      -0.127029    |    -1.44425   |      0.00601817  | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |            nan       |          nan       |
| 2026-03-10   | hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.154629   |      -0.107377    |    -4.25183   |      0.00990615  | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |            nan       |          nan       |
| 2026-03-10   | hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.178366   |      -0.127813    |    -4.02828   |     -7.44823e-05 | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |            nan       |          nan       |
| 2026-03-10   | hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.1353     |      -0.127029    |    -1.44425   |      0.00601817  | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |              5.02398 |           -8.52427 |
| 2026-03-10   | hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.154629   |      -0.107377    |    -4.25183   |      0.00990615  | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |              5.02398 |           -8.52427 |
| 2026-03-10   | hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.178366   |      -0.127813    |    -4.02828   |     -7.44823e-05 | 29.49 |          4.76229 |                  -1.93 |                0.330288 |                     0.427605 |              5.02398 |           -8.52427 |
| 2026-03-11   | heuristic                        |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0223529  |       0.0225122   |    -2.80758   |      0.00386472  | 25.5  |          4.80784 |                  -0.16 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-11   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.0498039  |      -0.000897227 |    -2.58402   |     -0.00605621  | 25.5  |          4.80784 |                  -0.16 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-11   | hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0223529  |       0.0225122   |    -2.80758   |      0.00386472  | 25.5  |          4.80784 |                  -0.16 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-11   | hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.0498039  |      -0.000897227 |    -2.58402   |     -0.00605621  | 25.5  |          4.80784 |                  -0.16 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-11   | hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0223529  |       0.0225122   |    -2.80758   |      0.00386472  | 25.5  |          4.80784 |                  -0.16 |                0.236237 |                     0.356404 |            nan       |          nan       |
| 2026-03-11   | hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.0498039  |      -0.000897227 |    -2.58402   |     -0.00605621  | 25.5  |          4.80784 |                  -0.16 |                0.236237 |                     0.356404 |            nan       |          nan       |
| 2026-03-11   | hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0223529  |       0.0225122   |    -2.80758   |      0.00386472  | 25.5  |          4.80784 |                  -0.16 |                0.236237 |                     0.356404 |              5.6002  |           -7.74345 |
| 2026-03-11   | hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.0498039  |      -0.000897227 |    -2.58402   |     -0.00605621  | 25.5  |          4.80784 |                  -0.16 |                0.236237 |                     0.356404 |              5.6002  |           -7.74345 |
| 2026-03-13   | heuristic                        |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0297152  |      -0.0466161   |    -0.576991  |     -0.00364991  | 24.23 |          5.0553  |                   0.74 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-13   | hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0297152  |      -0.0466161   |    -0.576991  |     -0.00364991  | 24.23 |          5.0553  |                   0.74 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-13   | hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0297152  |      -0.0466161   |    -0.576991  |     -0.00364991  | 24.23 |          5.0553  |                   0.74 |                0.262066 |                     0.387886 |            nan       |          nan       |
| 2026-03-13   | hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0297152  |      -0.0466161   |    -0.576991  |     -0.00364991  | 24.23 |          5.0553  |                   0.74 |                0.262066 |                     0.387886 |              5.27678 |           -8.34535 |
| 2026-03-16   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.138512   |      -0.102934    |    -0.710547  |      0.00348097  | 27.29 |          4.77025 |                  -0.34 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-16   | heuristic                        |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.180286   |      -0.150791    |    -0.673123  |      0.00561157  | 27.29 |          4.77025 |                  -0.34 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-16   | hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.138512   |      -0.102934    |    -0.710547  |      0.00348097  | 27.29 |          4.77025 |                  -0.34 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-16   | hmm_v1_core                      |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.180286   |      -0.150791    |    -0.673123  |      0.00561157  | 27.29 |          4.77025 |                  -0.34 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-16   | hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.138512   |      -0.102934    |    -0.710547  |      0.00348097  | 27.29 |          4.77025 |                  -0.34 |                0.249831 |                     0.381156 |            nan       |          nan       |
| 2026-03-16   | hmm_v2_core_plus_sector_corr     |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.180286   |      -0.150791    |    -0.673123  |      0.00561157  | 27.29 |          4.77025 |                  -0.34 |                0.249831 |                     0.381156 |            nan       |          nan       |
| 2026-03-16   | hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.138512   |      -0.102934    |    -0.710547  |      0.00348097  | 27.29 |          4.77025 |                  -0.34 |                0.249831 |                     0.381156 |              5.4321  |           -8.19754 |
| 2026-03-16   | hmm_v3_core_plus_sector_geometry |         3 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.180286   |      -0.150791    |    -0.673123  |      0.00561157  | 27.29 |          4.77025 |                  -0.34 |                0.249831 |                     0.381156 |              5.4321  |           -8.19754 |
| 2026-03-17   | heuristic                        |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.135344   |      -0.10889     |    -0.761215  |      0.00867205  | 27.19 |          4.81979 |                   0.09 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-17   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.177271   |      -0.156429    |    -0.723791  |      0.0108137   | 27.19 |          4.81979 |                   0.09 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-17   | hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.135344   |      -0.10889     |    -0.761215  |      0.00867205  | 27.19 |          4.81979 |                   0.09 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-17   | hmm_v1_core                      |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.177271   |      -0.156429    |    -0.723791  |      0.0108137   | 27.19 |          4.81979 |                   0.09 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-17   | hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.135344   |      -0.10889     |    -0.761215  |      0.00867205  | 27.19 |          4.81979 |                   0.09 |                0.238272 |                     0.388624 |            nan       |          nan       |
| 2026-03-17   | hmm_v2_core_plus_sector_corr     |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.177271   |      -0.156429    |    -0.723791  |      0.0108137   | 27.19 |          4.81979 |                   0.09 |                0.238272 |                     0.388624 |            nan       |          nan       |
| 2026-03-17   | hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.135344   |      -0.10889     |    -0.761215  |      0.00867205  | 27.19 |          4.81979 |                   0.09 |                0.238272 |                     0.388624 |              5.54896 |           -7.59146 |
| 2026-03-17   | hmm_v3_core_plus_sector_geometry |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.177271   |      -0.156429    |    -0.723791  |      0.0108137   | 27.19 |          4.81979 |                   0.09 |                0.238272 |                     0.388624 |              5.54896 |           -7.59146 |
| 2026-03-20   | heuristic                        |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0410522  |      -0.0664822   |    -0.274633  |      0.000287482 | 25.09 |          5.04185 |                   1.47 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-20   | hmm_v1_core                      |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0410522  |      -0.0664822   |    -0.274633  |      0.000287482 | 25.09 |          5.04185 |                   1.47 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-20   | hmm_v2_core_plus_sector_corr     |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0410522  |      -0.0664822   |    -0.274633  |      0.000287482 | 25.09 |          5.04185 |                   1.47 |                0.315204 |                     0.458806 |            nan       |          nan       |
| 2026-03-20   | hmm_v3_core_plus_sector_geometry |         1 | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP         |      -0.0410522  |      -0.0664822   |    -0.274633  |      0.000287482 | 25.09 |          5.04185 |                   1.47 |                0.315204 |                     0.458806 |              5.07223 |           -7.7102  |
| 2026-03-24   | heuristic                        |         1 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |      -0.023525   |      -0.0273994   |    -0.0641136 |      0.00531166  | 26.78 |          4.71546 |                   0.65 |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-24   | heuristic                        |         2 | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND |       0.00634802 |      -0.0169465   |    -0.280124  |      0.00653625  | 26.78 |          4.71546 |                   0.65 |              nan        |                   nan        |            nan       |          nan       |

## Missed Risks

| as_of_date   | model_name                       |   horizon | predicted_state      | realized_state           |   vix_change_pct |   vvix_change_pct |   rv21_change |   spy_return_pct |   vix |   vvix_vix_ratio |   term_structure_slope |   avg_pairwise_corr_21d |   first_eigenvalue_share_21d |   effective_rank_21d |   log_det_corr_21d |
|:-------------|:---------------------------------|----------:|:---------------------|:-------------------------|-----------------:|------------------:|--------------:|-----------------:|------:|-----------------:|-----------------------:|------------------------:|-----------------------------:|---------------------:|-------------------:|
| 2026-03-02   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0660225  |       0.0647144   |    0.00312972 |     -0.00375945  | 18.63 |          5.59045 |                   2.18 |                0.201333 |                     0.368234 |              6.09447 |           -6.03403 |
| 2026-03-02   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.150832   |       0.0891983   |   -0.00316342 |     -0.00269573  | 18.63 |          5.59045 |                   2.18 |                0.201333 |                     0.368234 |              6.09447 |           -6.03403 |
| 2026-03-02   | hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND | HIGH_VOL_RISK_OFF        |       0.265164   |       0.11397     |    0.13159    |     -0.00961721  | 18.63 |          5.59045 |                   2.18 |                0.201333 |                     0.368234 |              6.09447 |           -6.03403 |
| 2026-03-03   | hmm_v1_core                      |         2 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.186808   |       0.0462621   |    0.128461   |     -0.00587986  | 19.86 |          5.58359 |                   1.7  |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-03   | hmm_v1_core                      |         3 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.0649547  |      -0.0356209   |    0.232831   |      0.00473899  | 19.86 |          5.58359 |                   1.7  |              nan        |                   nan        |            nan       |          nan       |
| 2026-03-03   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.186808   |       0.0462621   |    0.128461   |     -0.00587986  | 19.86 |          5.58359 |                   1.7  |                0.20473  |                     0.363719 |              6.01875 |           -6.12046 |
| 2026-03-03   | hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0649547  |      -0.0356209   |    0.232831   |      0.00473899  | 19.86 |          5.58359 |                   1.7  |                0.20473  |                     0.363719 |              6.01875 |           -6.12046 |
| 2026-04-20   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0795195  |       0.031746    |    0.0570141  |     -0.00177278  | 17.48 |          5.44222 |                   3.03 |              nan        |                   nan        |            nan       |          nan       |
| 2026-04-20   | hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.115561   |       0.0710607   |   -0.581437   |     -0.00527612  | 17.48 |          5.44222 |                   3.03 |              nan        |                   nan        |            nan       |          nan       |
| 2026-04-20   | hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.0795195  |       0.031746    |    0.0570141  |     -0.00177278  | 17.48 |          5.44222 |                   3.03 |                0.17345  |                     0.491072 |            nan       |          nan       |
| 2026-04-20   | hmm_v2_core_plus_sector_corr     |         2 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.115561   |       0.0710607   |   -0.581437   |     -0.00527612  | 17.48 |          5.44222 |                   3.03 |                0.17345  |                     0.491072 |            nan       |          nan       |
| 2026-04-20   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0795195  |       0.031746    |    0.0570141  |     -0.00177278  | 17.48 |          5.44222 |                   3.03 |                0.17345  |                     0.491072 |              5.01725 |           -8.63628 |
| 2026-04-20   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.115561   |       0.0710607   |   -0.581437   |     -0.00527612  | 17.48 |          5.44222 |                   3.03 |                0.17345  |                     0.491072 |              5.01725 |           -8.63628 |
| 2026-04-23   | hmm_v1_core                      |         1 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.0206131  |      -0.00162058  |    0.0627648  |     -0.00125276  | 18.92 |          5.21829 |                   2.32 |              nan        |                   nan        |            nan       |          nan       |
| 2026-04-23   | hmm_v2_core_plus_sector_corr     |         1 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.0206131  |      -0.00162058  |    0.0627648  |     -0.00125276  | 18.92 |          5.21829 |                   2.32 |                0.159283 |                     0.513305 |            nan       |          nan       |
| 2026-04-23   | hmm_v3_core_plus_sector_geometry |         1 | MID_VOL_CHOP         | VOL_EXPANSION_TRANSITION |       0.0206131  |      -0.00162058  |    0.0627648  |     -0.00125276  | 18.92 |          5.21829 |                   2.32 |                0.159283 |                     0.513305 |              4.75801 |           -8.84962 |
| 2026-05-01   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00592066 |       0.0156884   |    0.117588   |      0.000180588 | 16.89 |          5.54766 |                   3.19 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-01   | hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0828893  |       0.0489861   |    0.420244   |     -0.00293108  | 16.89 |          5.54766 |                   3.19 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-01   | hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0290112  |       0.0166489   |    0.743553   |      0.00915443  | 16.89 |          5.54766 |                   3.19 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-01   | hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00592066 |       0.0156884   |    0.117588   |      0.000180588 | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |            nan       |          nan       |
| 2026-05-01   | hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0828893  |       0.0489861   |    0.420244   |     -0.00293108  | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |            nan       |          nan       |
| 2026-05-01   | hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0290112  |       0.0166489   |    0.743553   |      0.00915443  | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |            nan       |          nan       |
| 2026-05-01   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00592066 |       0.0156884   |    0.117588   |      0.000180588 | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |              4.89047 |           -8.86623 |
| 2026-05-01   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0828893  |       0.0489861   |    0.420244   |     -0.00293108  | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |              4.89047 |           -8.86623 |
| 2026-05-01   | hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0290112  |       0.0166489   |    0.743553   |      0.00915443  | 16.89 |          5.54766 |                   3.19 |                0.202176 |                     0.444802 |              4.89047 |           -8.86623 |
| 2026-05-04   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0765156  |       0.0327834   |    0.302656   |     -0.00311111  | 16.99 |          5.60153 |                   3.38 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-04   | hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0229547  |       0.000945676 |    0.625965   |      0.00897222  | 16.99 |          5.60153 |                   3.38 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-04   | hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0765156  |       0.0327834   |    0.302656   |     -0.00311111  | 16.99 |          5.60153 |                   3.38 |                0.225342 |                     0.4608   |            nan       |          nan       |
| 2026-05-04   | hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0229547  |       0.000945676 |    0.625965   |      0.00897222  | 16.99 |          5.60153 |                   3.38 |                0.225342 |                     0.4608   |            nan       |          nan       |
| 2026-05-04   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0765156  |       0.0327834   |    0.302656   |     -0.00311111  | 16.99 |          5.60153 |                   3.38 |                0.225342 |                     0.4608   |              4.72284 |           -9.18256 |
| 2026-05-04   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0229547  |       0.000945676 |    0.625965   |      0.00897222  | 16.99 |          5.60153 |                   3.38 |                0.225342 |                     0.4608   |              4.72284 |           -9.18256 |
| 2026-05-05   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.049754   |      -0.0308271   |    0.323309   |      0.012121    | 18.29 |          5.37397 |                   2.76 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-05   | hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.049754   |      -0.0308271   |    0.323309   |      0.012121    | 18.29 |          5.37397 |                   2.76 |                0.235115 |                     0.470613 |            nan       |          nan       |
| 2026-05-05   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.049754   |      -0.0308271   |    0.323309   |      0.012121    | 18.29 |          5.37397 |                   2.76 |                0.235115 |                     0.470613 |              4.63496 |           -9.46371 |
| 2026-05-07   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0178263  |      -0.000960512 |    0.226849   |     -0.00212824  | 17.39 |          5.38815 |                   3.18 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-07   | hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0115009  |       0.0328709   |    0.331872   |      0.00657572  | 17.39 |          5.38815 |                   3.18 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-07   | hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0569293  |       0.0465315   |    0.332916   |      0.00856753  | 17.39 |          5.38815 |                   3.18 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-07   | hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0178263  |      -0.000960512 |    0.226849   |     -0.00212824  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |            nan       |          nan       |
| 2026-05-07   | hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0115009  |       0.0328709   |    0.331872   |      0.00657572  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |            nan       |          nan       |
| 2026-05-07   | hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0569293  |       0.0465315   |    0.332916   |      0.00856753  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |            nan       |          nan       |
| 2026-05-07   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0178263  |      -0.000960512 |    0.226849   |     -0.00212824  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |              5.09973 |           -7.9903  |
| 2026-05-07   | hmm_v3_core_plus_sector_geometry |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |      -0.0115009  |       0.0328709   |    0.331872   |      0.00657572  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |              5.09973 |           -7.9903  |
| 2026-05-07   | hmm_v3_core_plus_sector_geometry |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0569293  |       0.0465315   |    0.332916   |      0.00856753  | 17.39 |          5.38815 |                   3.18 |                0.222853 |                     0.455948 |              5.09973 |           -7.9903  |
| 2026-05-08   | hmm_v1_core                      |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00644028 |       0.0338639   |    0.105023   |      0.00872252  | 17.08 |          5.48068 |                   3.27 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-08   | hmm_v1_core                      |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0761124  |       0.0475377   |    0.106068   |      0.0107186   | 17.08 |          5.48068 |                   3.27 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-08   | hmm_v1_core                      |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0532787  |       0.0527721   |    0.305128   |      0.00715028  | 17.08 |          5.48068 |                   3.27 |              nan        |                   nan        |            nan       |          nan       |
| 2026-05-08   | hmm_v2_core_plus_sector_corr     |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00644028 |       0.0338639   |    0.105023   |      0.00872252  | 17.08 |          5.48068 |                   3.27 |                0.247999 |                     0.450518 |            nan       |          nan       |
| 2026-05-08   | hmm_v2_core_plus_sector_corr     |         2 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0761124  |       0.0475377   |    0.106068   |      0.0107186   | 17.08 |          5.48068 |                   3.27 |                0.247999 |                     0.450518 |            nan       |          nan       |
| 2026-05-08   | hmm_v2_core_plus_sector_corr     |         3 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.0532787  |       0.0527721   |    0.305128   |      0.00715028  | 17.08 |          5.48068 |                   3.27 |                0.247999 |                     0.450518 |            nan       |          nan       |
| 2026-05-08   | hmm_v3_core_plus_sector_geometry |         1 | STABLE_LOW_VOL_TREND | VOL_EXPANSION_TRANSITION |       0.00644028 |       0.0338639   |    0.105023   |      0.00872252  | 17.08 |          5.48068 |                   3.27 |                0.247999 |                     0.450518 |              5.09518 |           -8.22768 |

## Recent As-of Date Comparison

| As Of      | Model                            | Predicted State          | T+1 Outcome              | T+2 Outcome              | T+3 Outcome              | Score   |
|:-----------|:---------------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:--------|
| 2026-06-01 | hmm_v2_core_plus_sector_corr     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-01 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-02 | heuristic                        | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-02 | hmm_v1_core                      | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-02 | hmm_v2_core_plus_sector_corr     | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-02 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        |         |
| 2026-06-03 | heuristic                        | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-03 | hmm_v1_core                      | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-03 | hmm_v2_core_plus_sector_corr     | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-03 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | VOL_EXPANSION_TRANSITION |         |
| 2026-06-04 | heuristic                        | VOL_EXPANSION_TRANSITION | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-04 | hmm_v1_core                      | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-04 | hmm_v2_core_plus_sector_corr     | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-04 | hmm_v3_core_plus_sector_geometry | STABLE_LOW_VOL_TREND     | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        | HIGH_VOL_RISK_OFF        |         |
| 2026-06-05 | heuristic                        | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |
| 2026-06-05 | hmm_v1_core                      | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |
| 2026-06-05 | hmm_v2_core_plus_sector_corr     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |
| 2026-06-05 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | MID_VOL_CHOP             | VOL_EXPANSION_TRANSITION |         |
| 2026-06-08 | heuristic                        | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-08 | hmm_v1_core                      | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-08 | hmm_v2_core_plus_sector_corr     | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-08 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | STABLE_LOW_VOL_TREND     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-09 | heuristic                        | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-09 | hmm_v1_core                      | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-09 | hmm_v2_core_plus_sector_corr     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-09 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |         |
| 2026-06-10 | heuristic                        | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |                          |         |
| 2026-06-10 | hmm_v1_core                      | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |                          |         |
| 2026-06-10 | hmm_v2_core_plus_sector_corr     | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |                          |         |
| 2026-06-10 | hmm_v3_core_plus_sector_geometry | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION | VOL_EXPANSION_TRANSITION |                          |         |

## Model Disagreement Cases

| as_of_date   | hmm_v3                   | other_model                  | other_state              |
|:-------------|:-------------------------|:-----------------------------|:-------------------------|
| 2026-03-02   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-03-02   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | VOL_EXPANSION_TRANSITION |
| 2026-03-02   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | VOL_EXPANSION_TRANSITION |
| 2026-03-03   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-03-03   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-03-03   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | VOL_EXPANSION_TRANSITION |
| 2026-03-10   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | HIGH_VOL_RISK_OFF        |
| 2026-03-12   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | HIGH_VOL_RISK_OFF        |
| 2026-03-31   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | HIGH_VOL_RISK_OFF        |
| 2026-04-01   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | HIGH_VOL_RISK_OFF        |
| 2026-04-02   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | HIGH_VOL_RISK_OFF        |
| 2026-04-09   | VOL_EXPANSION_TRANSITION | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-04-09   | VOL_EXPANSION_TRANSITION | hmm_v2_core_plus_sector_corr | MID_VOL_CHOP             |
| 2026-04-10   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-13   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-13   | MID_VOL_CHOP             | hmm_v1_core                  | STABLE_LOW_VOL_TREND     |
| 2026-04-14   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-15   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-16   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-17   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-17   | MID_VOL_CHOP             | hmm_v1_core                  | STABLE_LOW_VOL_TREND     |
| 2026-04-20   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-20   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | MID_VOL_CHOP             |
| 2026-04-21   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-21   | MID_VOL_CHOP             | hmm_v1_core                  | STABLE_LOW_VOL_TREND     |
| 2026-04-22   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-23   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-24   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-24   | MID_VOL_CHOP             | hmm_v1_core                  | STABLE_LOW_VOL_TREND     |
| 2026-04-27   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-27   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-04-28   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-29   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-04-29   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | MID_VOL_CHOP             |
| 2026-04-30   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-01   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-04   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-05   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-06   | MID_VOL_CHOP             | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-06   | MID_VOL_CHOP             | hmm_v1_core                  | STABLE_LOW_VOL_TREND     |
| 2026-05-06   | MID_VOL_CHOP             | hmm_v2_core_plus_sector_corr | STABLE_LOW_VOL_TREND     |
| 2026-05-07   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-08   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-11   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-12   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-13   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-14   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-15   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-18   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-18   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-05-18   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | MID_VOL_CHOP             |
| 2026-05-19   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-19   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-05-19   | STABLE_LOW_VOL_TREND     | hmm_v2_core_plus_sector_corr | MID_VOL_CHOP             |
| 2026-05-20   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-20   | STABLE_LOW_VOL_TREND     | hmm_v1_core                  | MID_VOL_CHOP             |
| 2026-05-21   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-22   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-25   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-26   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-27   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-28   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-05-29   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-06-01   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-06-02   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-06-03   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |
| 2026-06-04   | STABLE_LOW_VOL_TREND     | heuristic                    | VOL_EXPANSION_TRANSITION |

## HMM v3 Special Section

Track whether geometry features improved false vol-expansion avoidance and mid-vol chop detection.

## Model Usefulness Summary

- `heuristic` T+1: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `heuristic` T+1: false-alarm rate is elevated (46.6%); model may over-call expansion risk.
- `heuristic` T+1: Brier VIX Spike=0.4698, combined vol-direction accuracy=55.7%.
- `heuristic` T+2: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `heuristic` T+2: false-alarm rate is elevated (45.2%); model may over-call expansion risk.
- `heuristic` T+2: Brier VIX Spike=0.5208, combined vol-direction accuracy=52.1%.
- `heuristic` T+3: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `heuristic` T+3: false-alarm rate is elevated (44.4%); model may over-call expansion risk.
- `heuristic` T+3: Brier VIX Spike=0.5496, combined vol-direction accuracy=53.2%.
- `hmm_v1_core` T+1: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v1_core` T+1: missed-risk rate is elevated (27.4%); risk transitions may be under-warned.
- `hmm_v1_core` T+1: Brier VIX Spike=0.2721, combined vol-direction accuracy=63.9%.
- `hmm_v1_core` T+2: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v1_core` T+2: missed-risk rate is elevated (30.1%); risk transitions may be under-warned.
- `hmm_v1_core` T+2: Brier VIX Spike=0.2703, combined vol-direction accuracy=60.7%.
- `hmm_v1_core` T+3: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v1_core` T+3: missed-risk rate is elevated (26.4%); risk transitions may be under-warned.
- `hmm_v1_core` T+3: Brier VIX Spike=0.2231, combined vol-direction accuracy=61.1%.
- `hmm_v2_core_plus_sector_corr` T+1: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v2_core_plus_sector_corr` T+1: missed-risk rate is elevated (27.4%); risk transitions may be under-warned.
- `hmm_v2_core_plus_sector_corr` T+1: Brier VIX Spike=0.2889, combined vol-direction accuracy=64.8%.
- `hmm_v2_core_plus_sector_corr` T+2: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v2_core_plus_sector_corr` T+2: missed-risk rate is elevated (28.8%); risk transitions may be under-warned.
- `hmm_v2_core_plus_sector_corr` T+2: Brier VIX Spike=0.2643, combined vol-direction accuracy=61.6%.
- `hmm_v2_core_plus_sector_corr` T+3: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v2_core_plus_sector_corr` T+3: missed-risk rate is elevated (25.0%); risk transitions may be under-warned.
- `hmm_v2_core_plus_sector_corr` T+3: Brier VIX Spike=0.2210, combined vol-direction accuracy=61.6%.
- `hmm_v3_core_plus_sector_geometry` T+1: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v3_core_plus_sector_geometry` T+1: missed-risk rate is elevated (27.4%); risk transitions may be under-warned.
- `hmm_v3_core_plus_sector_geometry` T+1: Brier VIX Spike=0.2697, combined vol-direction accuracy=62.1%.
- `hmm_v3_core_plus_sector_geometry` T+2: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v3_core_plus_sector_geometry` T+2: missed-risk rate is elevated (31.5%); risk transitions may be under-warned.
- `hmm_v3_core_plus_sector_geometry` T+2: Brier VIX Spike=0.2785, combined vol-direction accuracy=58.4%.
- `hmm_v3_core_plus_sector_geometry` T+3: exact accuracy is lower than adjacent-tolerant accuracy, so most misses are neighboring-regime errors.
- `hmm_v3_core_plus_sector_geometry` T+3: missed-risk rate is elevated (27.8%); risk transitions may be under-warned.
- `hmm_v3_core_plus_sector_geometry` T+3: Brier VIX Spike=0.2151, combined vol-direction accuracy=58.8%.

## Diagnostics

| as_of_date   | model_name                       | converged   |   training_row_count | training_end_date   | warnings   |
|:-------------|:---------------------------------|:------------|---------------------:|:--------------------|:-----------|
| 2026-03-02   | heuristic                        | True        |                  756 | 2026-03-02          |            |
| 2026-03-02   | hmm_v1_core                      | True        |                  756 | 2026-03-02          |            |
| 2026-03-02   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-02          |            |
| 2026-03-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-02          |            |
| 2026-03-03   | heuristic                        | True        |                  756 | 2026-03-03          |            |
| 2026-03-03   | hmm_v1_core                      | True        |                  756 | 2026-03-03          |            |
| 2026-03-03   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-03          |            |
| 2026-03-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-03          |            |
| 2026-03-04   | heuristic                        | True        |                  756 | 2026-03-04          |            |
| 2026-03-04   | hmm_v1_core                      | True        |                  756 | 2026-03-04          |            |
| 2026-03-04   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-04          |            |
| 2026-03-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-04          |            |
| 2026-03-05   | heuristic                        | True        |                  756 | 2026-03-05          |            |
| 2026-03-05   | hmm_v1_core                      | True        |                  756 | 2026-03-05          |            |
| 2026-03-05   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-05          |            |
| 2026-03-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-05          |            |
| 2026-03-06   | heuristic                        | True        |                  756 | 2026-03-06          |            |
| 2026-03-06   | hmm_v1_core                      | True        |                  756 | 2026-03-06          |            |
| 2026-03-06   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-06          |            |
| 2026-03-06   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-06          |            |
| 2026-03-09   | heuristic                        | True        |                  756 | 2026-03-09          |            |
| 2026-03-09   | hmm_v1_core                      | True        |                  756 | 2026-03-09          |            |
| 2026-03-09   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-09          |            |
| 2026-03-09   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-09          |            |
| 2026-03-10   | heuristic                        | True        |                  756 | 2026-03-10          |            |
| 2026-03-10   | hmm_v1_core                      | True        |                  756 | 2026-03-10          |            |
| 2026-03-10   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-10          |            |
| 2026-03-10   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-10          |            |
| 2026-03-11   | heuristic                        | True        |                  756 | 2026-03-11          |            |
| 2026-03-11   | hmm_v1_core                      | True        |                  756 | 2026-03-11          |            |
| 2026-03-11   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-11          |            |
| 2026-03-11   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-11          |            |
| 2026-03-12   | heuristic                        | True        |                  756 | 2026-03-12          |            |
| 2026-03-12   | hmm_v1_core                      | True        |                  756 | 2026-03-12          |            |
| 2026-03-12   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-12          |            |
| 2026-03-12   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-12          |            |
| 2026-03-13   | heuristic                        | True        |                  756 | 2026-03-13          |            |
| 2026-03-13   | hmm_v1_core                      | True        |                  756 | 2026-03-13          |            |
| 2026-03-13   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-13          |            |
| 2026-03-13   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-13          |            |
| 2026-03-16   | heuristic                        | True        |                  756 | 2026-03-16          |            |
| 2026-03-16   | hmm_v1_core                      | True        |                  756 | 2026-03-16          |            |
| 2026-03-16   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-16          |            |
| 2026-03-16   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-16          |            |
| 2026-03-17   | heuristic                        | True        |                  756 | 2026-03-17          |            |
| 2026-03-17   | hmm_v1_core                      | True        |                  756 | 2026-03-17          |            |
| 2026-03-17   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-17          |            |
| 2026-03-17   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-17          |            |
| 2026-03-18   | heuristic                        | True        |                  756 | 2026-03-18          |            |
| 2026-03-18   | hmm_v1_core                      | True        |                  756 | 2026-03-18          |            |
| 2026-03-18   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-18          |            |
| 2026-03-18   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-18          |            |
| 2026-03-19   | heuristic                        | True        |                  756 | 2026-03-19          |            |
| 2026-03-19   | hmm_v1_core                      | True        |                  756 | 2026-03-19          |            |
| 2026-03-19   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-19          |            |
| 2026-03-19   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-19          |            |
| 2026-03-20   | heuristic                        | True        |                  756 | 2026-03-20          |            |
| 2026-03-20   | hmm_v1_core                      | True        |                  756 | 2026-03-20          |            |
| 2026-03-20   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-20          |            |
| 2026-03-20   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-20          |            |
| 2026-03-23   | heuristic                        | True        |                  756 | 2026-03-23          |            |
| 2026-03-23   | hmm_v1_core                      | True        |                  756 | 2026-03-23          |            |
| 2026-03-23   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-23          |            |
| 2026-03-23   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-23          |            |
| 2026-03-24   | heuristic                        | True        |                  756 | 2026-03-24          |            |
| 2026-03-24   | hmm_v1_core                      | True        |                  756 | 2026-03-24          |            |
| 2026-03-24   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-24          |            |
| 2026-03-24   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-24          |            |
| 2026-03-25   | heuristic                        | True        |                  756 | 2026-03-25          |            |
| 2026-03-25   | hmm_v1_core                      | True        |                  756 | 2026-03-25          |            |
| 2026-03-25   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-25          |            |
| 2026-03-25   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-25          |            |
| 2026-03-26   | heuristic                        | True        |                  756 | 2026-03-26          |            |
| 2026-03-26   | hmm_v1_core                      | True        |                  756 | 2026-03-26          |            |
| 2026-03-26   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-26          |            |
| 2026-03-26   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-26          |            |
| 2026-03-27   | heuristic                        | True        |                  756 | 2026-03-27          |            |
| 2026-03-27   | hmm_v1_core                      | True        |                  756 | 2026-03-27          |            |
| 2026-03-27   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-27          |            |
| 2026-03-27   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-27          |            |
| 2026-03-30   | heuristic                        | True        |                  756 | 2026-03-30          |            |
| 2026-03-30   | hmm_v1_core                      | True        |                  756 | 2026-03-30          |            |
| 2026-03-30   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-30          |            |
| 2026-03-30   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-30          |            |
| 2026-03-31   | heuristic                        | True        |                  756 | 2026-03-31          |            |
| 2026-03-31   | hmm_v1_core                      | True        |                  756 | 2026-03-31          |            |
| 2026-03-31   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-03-31          |            |
| 2026-03-31   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-03-31          |            |
| 2026-04-01   | heuristic                        | True        |                  756 | 2026-04-01          |            |
| 2026-04-01   | hmm_v1_core                      | True        |                  756 | 2026-04-01          |            |
| 2026-04-01   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-01          |            |
| 2026-04-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-01          |            |
| 2026-04-02   | heuristic                        | True        |                  756 | 2026-04-02          |            |
| 2026-04-02   | hmm_v1_core                      | True        |                  756 | 2026-04-02          |            |
| 2026-04-02   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-02          |            |
| 2026-04-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-02          |            |
| 2026-04-03   | heuristic                        | True        |                  756 | 2026-04-03          |            |
| 2026-04-03   | hmm_v1_core                      | True        |                  756 | 2026-04-03          |            |
| 2026-04-03   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-03          |            |
| 2026-04-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-03          |            |
| 2026-04-06   | heuristic                        | True        |                  756 | 2026-04-06          |            |
| 2026-04-06   | hmm_v1_core                      | True        |                  756 | 2026-04-06          |            |
| 2026-04-06   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-06          |            |
| 2026-04-06   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-06          |            |
| 2026-04-07   | heuristic                        | True        |                  756 | 2026-04-07          |            |
| 2026-04-07   | hmm_v1_core                      | True        |                  756 | 2026-04-07          |            |
| 2026-04-07   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-07          |            |
| 2026-04-07   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-07          |            |
| 2026-04-08   | heuristic                        | True        |                  756 | 2026-04-08          |            |
| 2026-04-08   | hmm_v1_core                      | True        |                  756 | 2026-04-08          |            |
| 2026-04-08   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-08          |            |
| 2026-04-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-08          |            |
| 2026-04-09   | heuristic                        | True        |                  756 | 2026-04-09          |            |
| 2026-04-09   | hmm_v1_core                      | True        |                  756 | 2026-04-09          |            |
| 2026-04-09   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-09          |            |
| 2026-04-09   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-09          |            |
| 2026-04-10   | heuristic                        | True        |                  756 | 2026-04-10          |            |
| 2026-04-10   | hmm_v1_core                      | True        |                  756 | 2026-04-10          |            |
| 2026-04-10   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-10          |            |
| 2026-04-10   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-10          |            |
| 2026-04-13   | heuristic                        | True        |                  756 | 2026-04-13          |            |
| 2026-04-13   | hmm_v1_core                      | True        |                  756 | 2026-04-13          |            |
| 2026-04-13   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-13          |            |
| 2026-04-13   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-13          |            |
| 2026-04-14   | heuristic                        | True        |                  756 | 2026-04-14          |            |
| 2026-04-14   | hmm_v1_core                      | True        |                  756 | 2026-04-14          |            |
| 2026-04-14   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-14          |            |
| 2026-04-14   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-14          |            |
| 2026-04-15   | heuristic                        | True        |                  756 | 2026-04-15          |            |
| 2026-04-15   | hmm_v1_core                      | True        |                  756 | 2026-04-15          |            |
| 2026-04-15   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-15          |            |
| 2026-04-15   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-15          |            |
| 2026-04-16   | heuristic                        | True        |                  756 | 2026-04-16          |            |
| 2026-04-16   | hmm_v1_core                      | True        |                  756 | 2026-04-16          |            |
| 2026-04-16   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-16          |            |
| 2026-04-16   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-16          |            |
| 2026-04-17   | heuristic                        | True        |                  756 | 2026-04-17          |            |
| 2026-04-17   | hmm_v1_core                      | True        |                  756 | 2026-04-17          |            |
| 2026-04-17   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-17          |            |
| 2026-04-17   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-17          |            |
| 2026-04-20   | heuristic                        | True        |                  756 | 2026-04-20          |            |
| 2026-04-20   | hmm_v1_core                      | True        |                  756 | 2026-04-20          |            |
| 2026-04-20   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-20          |            |
| 2026-04-20   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-20          |            |
| 2026-04-21   | heuristic                        | True        |                  756 | 2026-04-21          |            |
| 2026-04-21   | hmm_v1_core                      | True        |                  756 | 2026-04-21          |            |
| 2026-04-21   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-21          |            |
| 2026-04-21   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-21          |            |
| 2026-04-22   | heuristic                        | True        |                  756 | 2026-04-22          |            |
| 2026-04-22   | hmm_v1_core                      | True        |                  756 | 2026-04-22          |            |
| 2026-04-22   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-22          |            |
| 2026-04-22   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-22          |            |
| 2026-04-23   | heuristic                        | True        |                  756 | 2026-04-23          |            |
| 2026-04-23   | hmm_v1_core                      | True        |                  756 | 2026-04-23          |            |
| 2026-04-23   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-23          |            |
| 2026-04-23   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-23          |            |
| 2026-04-24   | heuristic                        | True        |                  756 | 2026-04-24          |            |
| 2026-04-24   | hmm_v1_core                      | True        |                  756 | 2026-04-24          |            |
| 2026-04-24   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-24          |            |
| 2026-04-24   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-24          |            |
| 2026-04-27   | heuristic                        | True        |                  756 | 2026-04-27          |            |
| 2026-04-27   | hmm_v1_core                      | True        |                  756 | 2026-04-27          |            |
| 2026-04-27   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-27          |            |
| 2026-04-27   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-27          |            |
| 2026-04-28   | heuristic                        | True        |                  756 | 2026-04-28          |            |
| 2026-04-28   | hmm_v1_core                      | True        |                  756 | 2026-04-28          |            |
| 2026-04-28   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-28          |            |
| 2026-04-28   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-28          |            |
| 2026-04-29   | heuristic                        | True        |                  756 | 2026-04-29          |            |
| 2026-04-29   | hmm_v1_core                      | True        |                  756 | 2026-04-29          |            |
| 2026-04-29   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-29          |            |
| 2026-04-29   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-29          |            |
| 2026-04-30   | heuristic                        | True        |                  756 | 2026-04-30          |            |
| 2026-04-30   | hmm_v1_core                      | True        |                  756 | 2026-04-30          |            |
| 2026-04-30   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-04-30          |            |
| 2026-04-30   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-04-30          |            |
| 2026-05-01   | heuristic                        | True        |                  756 | 2026-05-01          |            |
| 2026-05-01   | hmm_v1_core                      | True        |                  756 | 2026-05-01          |            |
| 2026-05-01   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-01          |            |
| 2026-05-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-01          |            |
| 2026-05-04   | heuristic                        | True        |                  756 | 2026-05-04          |            |
| 2026-05-04   | hmm_v1_core                      | True        |                  756 | 2026-05-04          |            |
| 2026-05-04   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-04          |            |
| 2026-05-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-04          |            |
| 2026-05-05   | heuristic                        | True        |                  756 | 2026-05-05          |            |
| 2026-05-05   | hmm_v1_core                      | True        |                  756 | 2026-05-05          |            |
| 2026-05-05   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-05          |            |
| 2026-05-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-05          |            |
| 2026-05-06   | heuristic                        | True        |                  756 | 2026-05-06          |            |
| 2026-05-06   | hmm_v1_core                      | True        |                  756 | 2026-05-06          |            |
| 2026-05-06   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-06          |            |
| 2026-05-06   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-06          |            |
| 2026-05-07   | heuristic                        | True        |                  756 | 2026-05-07          |            |
| 2026-05-07   | hmm_v1_core                      | True        |                  756 | 2026-05-07          |            |
| 2026-05-07   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-07          |            |
| 2026-05-07   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-07          |            |
| 2026-05-08   | heuristic                        | True        |                  756 | 2026-05-08          |            |
| 2026-05-08   | hmm_v1_core                      | True        |                  756 | 2026-05-08          |            |
| 2026-05-08   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-08          |            |
| 2026-05-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-08          |            |
| 2026-05-11   | heuristic                        | True        |                  756 | 2026-05-11          |            |
| 2026-05-11   | hmm_v1_core                      | True        |                  756 | 2026-05-11          |            |
| 2026-05-11   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-11          |            |
| 2026-05-11   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-11          |            |
| 2026-05-12   | heuristic                        | True        |                  756 | 2026-05-12          |            |
| 2026-05-12   | hmm_v1_core                      | True        |                  756 | 2026-05-12          |            |
| 2026-05-12   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-12          |            |
| 2026-05-12   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-12          |            |
| 2026-05-13   | heuristic                        | True        |                  756 | 2026-05-13          |            |
| 2026-05-13   | hmm_v1_core                      | True        |                  756 | 2026-05-13          |            |
| 2026-05-13   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-13          |            |
| 2026-05-13   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-13          |            |
| 2026-05-14   | heuristic                        | True        |                  756 | 2026-05-14          |            |
| 2026-05-14   | hmm_v1_core                      | True        |                  756 | 2026-05-14          |            |
| 2026-05-14   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-14          |            |
| 2026-05-14   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-14          |            |
| 2026-05-15   | heuristic                        | True        |                  756 | 2026-05-15          |            |
| 2026-05-15   | hmm_v1_core                      | True        |                  756 | 2026-05-15          |            |
| 2026-05-15   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-15          |            |
| 2026-05-15   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-15          |            |
| 2026-05-18   | heuristic                        | True        |                  756 | 2026-05-18          |            |
| 2026-05-18   | hmm_v1_core                      | True        |                  756 | 2026-05-18          |            |
| 2026-05-18   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-18          |            |
| 2026-05-18   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-18          |            |
| 2026-05-19   | heuristic                        | True        |                  756 | 2026-05-19          |            |
| 2026-05-19   | hmm_v1_core                      | True        |                  756 | 2026-05-19          |            |
| 2026-05-19   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-19          |            |
| 2026-05-19   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-19          |            |
| 2026-05-20   | heuristic                        | True        |                  756 | 2026-05-20          |            |
| 2026-05-20   | hmm_v1_core                      | True        |                  756 | 2026-05-20          |            |
| 2026-05-20   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-20          |            |
| 2026-05-20   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-20          |            |
| 2026-05-21   | heuristic                        | True        |                  756 | 2026-05-21          |            |
| 2026-05-21   | hmm_v1_core                      | True        |                  756 | 2026-05-21          |            |
| 2026-05-21   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-21          |            |
| 2026-05-21   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-21          |            |
| 2026-05-22   | heuristic                        | True        |                  756 | 2026-05-22          |            |
| 2026-05-22   | hmm_v1_core                      | True        |                  756 | 2026-05-22          |            |
| 2026-05-22   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-22          |            |
| 2026-05-22   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-22          |            |
| 2026-05-25   | heuristic                        | True        |                  756 | 2026-05-25          |            |
| 2026-05-25   | hmm_v1_core                      | True        |                  756 | 2026-05-25          |            |
| 2026-05-25   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-25          |            |
| 2026-05-25   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-25          |            |
| 2026-05-26   | heuristic                        | True        |                  756 | 2026-05-26          |            |
| 2026-05-26   | hmm_v1_core                      | True        |                  756 | 2026-05-26          |            |
| 2026-05-26   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-26          |            |
| 2026-05-26   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-26          |            |
| 2026-05-27   | heuristic                        | True        |                  756 | 2026-05-27          |            |
| 2026-05-27   | hmm_v1_core                      | True        |                  756 | 2026-05-27          |            |
| 2026-05-27   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-27          |            |
| 2026-05-27   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-27          |            |
| 2026-05-28   | heuristic                        | True        |                  756 | 2026-05-28          |            |
| 2026-05-28   | hmm_v1_core                      | True        |                  756 | 2026-05-28          |            |
| 2026-05-28   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-28          |            |
| 2026-05-28   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-28          |            |
| 2026-05-29   | heuristic                        | True        |                  756 | 2026-05-29          |            |
| 2026-05-29   | hmm_v1_core                      | True        |                  756 | 2026-05-29          |            |
| 2026-05-29   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-05-29          |            |
| 2026-05-29   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-05-29          |            |
| 2026-06-01   | heuristic                        | True        |                  756 | 2026-06-01          |            |
| 2026-06-01   | hmm_v1_core                      | True        |                  756 | 2026-06-01          |            |
| 2026-06-01   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-01          |            |
| 2026-06-01   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-01          |            |
| 2026-06-02   | heuristic                        | True        |                  756 | 2026-06-02          |            |
| 2026-06-02   | hmm_v1_core                      | True        |                  756 | 2026-06-02          |            |
| 2026-06-02   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-02          |            |
| 2026-06-02   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-02          |            |
| 2026-06-03   | heuristic                        | True        |                  756 | 2026-06-03          |            |
| 2026-06-03   | hmm_v1_core                      | True        |                  756 | 2026-06-03          |            |
| 2026-06-03   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-03          |            |
| 2026-06-03   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-03          |            |
| 2026-06-04   | heuristic                        | True        |                  756 | 2026-06-04          |            |
| 2026-06-04   | hmm_v1_core                      | True        |                  756 | 2026-06-04          |            |
| 2026-06-04   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-04          |            |
| 2026-06-04   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-04          |            |
| 2026-06-05   | heuristic                        | True        |                  756 | 2026-06-05          |            |
| 2026-06-05   | hmm_v1_core                      | True        |                  756 | 2026-06-05          |            |
| 2026-06-05   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-05          |            |
| 2026-06-05   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-05          |            |
| 2026-06-08   | heuristic                        | True        |                  756 | 2026-06-08          |            |
| 2026-06-08   | hmm_v1_core                      | True        |                  756 | 2026-06-08          |            |
| 2026-06-08   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-08          |            |
| 2026-06-08   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-08          |            |
| 2026-06-09   | heuristic                        | True        |                  756 | 2026-06-09          |            |
| 2026-06-09   | hmm_v1_core                      | True        |                  756 | 2026-06-09          |            |
| 2026-06-09   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-09          |            |
| 2026-06-09   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-09          |            |
| 2026-06-10   | heuristic                        | True        |                  756 | 2026-06-10          |            |
| 2026-06-10   | hmm_v1_core                      | True        |                  756 | 2026-06-10          |            |
| 2026-06-10   | hmm_v2_core_plus_sector_corr     | True        |                  756 | 2026-06-10          |            |
| 2026-06-10   | hmm_v3_core_plus_sector_geometry | True        |                  756 | 2026-06-10          |            |
