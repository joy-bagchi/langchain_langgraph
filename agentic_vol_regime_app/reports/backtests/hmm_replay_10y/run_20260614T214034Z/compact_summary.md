# HMM Replay Compact Summary

- Run mode: `tuning`
- Models: `hmm_v4_path_aware_meta, hmm_v3_1_meta_blend`
- Horizons: `1, 3`
- Replay dates evaluated: `75`
- Prediction records: `150`
- Scored records: `300`
- Cache hits: `0`
- Cache misses: `150`
- Peak RSS memory (MB): `304.617`

## Runtime Profile (seconds)
- feature_load_seconds: `0.083065`
- feature_engineering_seconds: `3.527`
- model_fit_seconds: `11.557172`
- prediction_seconds: `11.557091`
- scoring_seconds: `0.004623`
- report_generation_seconds: `0.034201`
- total_seconds: `15.505022`

## Summary Metrics Preview

| model_name             |   horizon |   accuracy |   adjacent_tolerant_accuracy |   severe_miss_rate |   brier_vol_expansion |   brier_vix_spike |   risk_bucket_accuracy |   false_alarm_rate |   missed_risk_rate |   vix_directional_accuracy |   vvix_directional_accuracy |   rv_directional_accuracy |   combined_vol_directional_accuracy |   avg_lead_quality | notes   |
|:-----------------------|----------:|-----------:|-----------------------------:|-------------------:|----------------------:|------------------:|-----------------------:|-------------------:|-------------------:|---------------------------:|----------------------------:|--------------------------:|------------------------------------:|-------------------:|:--------|
| hmm_v3_1_meta_blend    |         1 |   0.364865 |                     0.837838 |           0.162162 |              0.360702 |          0.279847 |               0.635135 |          0.162162  |           0.202703 |                   0.594595 |                    0.608108 |                  0.77027  |                            0.657658 |           0.554054 |         |
| hmm_v3_1_meta_blend    |         3 |   0.361111 |                     0.805556 |           0.194444 |              0.308911 |          0.255602 |               0.75     |          0.0833333 |           0.166667 |                   0.583333 |                    0.569444 |                  0.805556 |                            0.652778 |           0.625    |         |
| hmm_v4_path_aware_meta |         1 |   0.202703 |                     0.513514 |           0.486486 |              0.675676 |          0.108108 |               0.513514 |          0         |           0.486486 |                   0.648649 |                    0.621622 |                  0.743243 |                            0.671171 |           0.554054 |         |
| hmm_v4_path_aware_meta |         3 |   0.416667 |                     0.486111 |           0.513889 |              0.722222 |          0.138889 |               0.486111 |          0         |           0.513889 |                   0.541667 |                    0.555556 |                  0.555556 |                            0.550926 |           0.625    |         |

_Tuning mode is intended for fast behavior discovery only. Use testing mode for final validation evidence._
