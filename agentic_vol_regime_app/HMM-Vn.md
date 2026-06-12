Update HMM architecture to support HMM v1, HMM v2, and HMM v3 as separate model configurations.

Do not modify HMM v1 behavior. HMM v1 is the current production baseline and should remain unchanged.

Goal:
Add experimental HMM v2 and HMM v3 variants that build additively on HMM v1.

Model philosophy:
- HMM v1 = vol-market lens
- HMM v2 = vol-market lens + sector-correlation lens
- HMM v3 = vol-market lens + sector-correlation lens + geometric dimensional-collapse lens

Do not replace vol features with sector/geometry features.
Sector correlation and wedge/Gram features are complementary, not substitutes.

Create model configs:

1. hmm_v1_core.yaml
   Existing production HMM feature set.
   Do not change.

2. hmm_v2_core_plus_sector_corr.yaml
   Use all HMM v1 features plus:
   - avg_pairwise_corr_21d
   - first_eigenvalue_share_21d

3. hmm_v3_core_plus_sector_geometry.yaml
   Use all HMM v1 features plus:
   - avg_pairwise_corr_21d
   - first_eigenvalue_share_21d
   - effective_rank_21d
   - log_det_corr_21d

Optional diagnostic-only model:
4. hmm_sector_geometry_only.yaml
   Use only sector-correlation / geometry features.
   This is for research diagnostics only.
   Do not use for production posture, DTE, or strike recommendation.

Sector universe:
- XLK
- XLF
- XLE
- XLY
- XLP
- XLI
- XLB
- XLV
- XLU
- XLRE

Feature calculations:

1. avg_pairwise_corr_21d
   - Compute daily returns for sector ETFs.
   - Compute trailing 21-day correlation matrix.
   - Average only off-diagonal pairwise correlations.

2. first_eigenvalue_share_21d
   - Use the trailing 21-day sector correlation matrix.
   - Compute eigenvalues using np.linalg.eigvalsh.
   - first_eigenvalue_share = largest_eigenvalue / sum(all_eigenvalues).
   - This measures market-mode dominance.

3. effective_rank_21d
   - Use eigenvalues of the sector correlation matrix.
   - Normalize eigenvalues into shares.
   - Compute entropy-based effective rank:
     effective_rank = exp(-sum(p_i * log(p_i)))
   - Lower effective rank means market dimensionality is collapsing.

4. log_det_corr_21d
   - Use the sector correlation matrix.
   - Apply small diagonal regularization if needed:
     C_reg = C + epsilon * I
   - Compute log determinant using np.linalg.slogdet.
   - More negative values indicate geometric volume collapse.
   - This approximates wedge-volume / Gram determinant behavior.

Important:
- Use correlation matrix first, not covariance matrix.
- Do not feed the full 10x10 matrix into HMM.
- Feed only compressed scalar features.
- Do not silently impute missing values with zero.
- If insufficient data exists, return explicit warnings and fall back to HMM v1.

Report changes:

Add section:
"Model Variant Comparison"

Include:

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---:|---:|---:|---:|---|
| HMM v1 Core | ... | ... | ... | ... | ... |
| HMM v2 Core + Sector Corr | ... | ... | ... | ... | ... |
| HMM v3 Core + Geometry | ... | ... | ... | ... | ... |

Add section:
"Sector Correlation / Market Mode"

Include:
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d if v3 enabled
- log_det_corr_21d if v3 enabled
- interpretation

Interpretation guide:
- Low avg correlation + low first eigenvalue share = stable sector independence.
- Rising avg correlation + rising first eigenvalue share = vol expansion risk.
- High first eigenvalue share + falling effective rank = one-factor market behavior.
- Falling log determinant = geometric volume collapse / dimensional compression.
- Vol explosion regimes should generally show highest market-mode dominance and lowest effective rank.

Diagnostics:

For each HMM variant, output:
- model convergence status
- state usage counts
- transition matrix
- expected state durations
- state feature means
- state ranking by VIX
- state ranking by first_eigenvalue_share if available
- state ranking by effective_rank if available
- warning if any state has <5% usage

Promotion rule:
- HMM v1 remains production baseline.
- HMM v2 is experimental until it improves state stability, transition prediction, or overwrite/DTE decisions.
- HMM v3 is experimental until separately validated.
- Do not let v2/v3 override production policy unless explicitly enabled by config.

Feature flags:
- enable_hmm_v2_sector_corr: true/false
- enable_hmm_v3_sector_geometry: true/false
- enable_sector_geometry_only_diagnostic: true/false

Tests:
Add tests for:
1. off-diagonal average correlation calculation
2. first_eigenvalue_share is between 0 and 1
3. perfectly correlated synthetic sectors produce high first_eigenvalue_share
4. independent synthetic sectors produce lower first_eigenvalue_share
5. effective_rank decreases when sectors become highly correlated
6. log_det_corr falls sharply when correlation matrix becomes near singular
7. missing sector symbols return warnings
8. insufficient history falls back to HMM v1
9. report renders model comparison section
10. historical replay does not use future sector data

Implementation order:

Tomorrow / HMM v2:
1. Add sector ETF data ingestion.
2. Add avg_pairwise_corr_21d.
3. Add first_eigenvalue_share_21d.
4. Create hmm_v2_core_plus_sector_corr.yaml.
5. Run HMM v1 and v2 side by side.
6. Add model comparison to Daily Belief Report.

Day after / HMM v3:
1. Add effective_rank_21d.
2. Add log_det_corr_21d.
3. Create hmm_v3_core_plus_sector_geometry.yaml.
4. Add geometry diagnostics.
5. Compare v1/v2/v3 in report.
6. Keep v3 experimental.

Do not implement Bayesian HMM, MCMC, RL, or automatic model promotion yet.