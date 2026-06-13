HMM v3 Behavior Change: Make Geometry a Smooth Modifier, Not a Hard Regime Gate

Context:
Current HMM v3 appears to let sector geometry dominate regime classification too aggressively. Geometry often downgrades VOL_EXPANSION into STABLE_LOW_VOL or MID_VOL_CHOP. This creates useful false-alarm filtering sometimes, but also too many false suppressions / missed risks.

Goal:
Change HMM v3 behavior so geometry acts as a smooth market-structure stress modifier rather than a hard veto over vol-market signals.

Do not delete HMM v3.
Do not modify HMM v1.
Do not change live trading behavior.
Create a new experimental model/layer:

- hmm_v3_1_meta_blend

Core philosophy:
HMM v1 remains the primary volatility-state model.
Geometry should modulate confidence/severity, not fully override vol risk.

Architecture:

1. HMM v1 Core Vol Model
   Produces:
   - core_vol_state
   - core_vol_state_probabilities
   - core_vol_risk_score

2. GeometryStressAgent
   Produces:
   - geometry_stress_score from 0.0 to 1.0
   - geometry_confirmation_level
   - geometry_rationale

3. MetaBlendRegimeAgent
   Combines:
   - HMM v1 core volatility risk
   - smooth geometry stress score
   into:
   - final_risk_score
   - final_regime
   - confidence_adjustment
   - recommended posture modifier

Do not feed geometry directly into the same HMM as equal-power features for this model.

GeometryStressAgent:

Create files:
- src/agents/geometry_stress_agent.py
- src/regime/meta_blend.py
- configs/models/hmm_v3_1_meta_blend.yaml
- tests/test_geometry_stress_agent.py
- tests/test_meta_blend_regime.py

Geometry features:
Use existing features:
- avg_pairwise_corr_21d
- first_eigenvalue_share_21d
- effective_rank_21d
- log_det_corr_21d

Convert each to a smooth 0-1 stress component.

Recommended transforms:

1. avg_corr_stress
   percentile rank of avg_pairwise_corr_21d over trailing lookback window.
   Higher avg correlation = higher stress.

2. eigen_stress
   percentile rank of first_eigenvalue_share_21d.
   Higher first eigenvalue share = higher one-factor stress.

3. effective_rank_stress
   percentile rank of negative effective_rank_21d.
   Lower effective rank = higher dimensional-collapse stress.

4. log_det_stress
   percentile rank of negative log_det_corr_21d.
   More negative log determinant = higher wedge/volume-collapse stress.

Default lookback:
- 252 trading days
Fallback:
- 126 trading days if 252 unavailable
Minimum:
- 63 trading days

If insufficient history, return warning and neutral geometry_stress_score = 0.50.

Geometry stress score:

geometry_stress_score =
  0.30 * avg_corr_stress
+ 0.30 * eigen_stress
+ 0.25 * effective_rank_stress
+ 0.15 * log_det_stress

All components must be clipped to [0, 1].

Geometry confirmation levels:
- 0.00 to 0.30: geometry_not_confirming
- 0.30 to 0.55: geometry_mild_confirmation
- 0.55 to 0.75: geometry_confirming
- 0.75 to 1.00: geometry_strong_confirmation

Core vol risk score:
Convert HMM v1 regime probabilities into severity-weighted risk score.

Severity mapping:
STABLE_LOW_VOL = 0.00
MID_VOL_CHOP = 0.33
VOL_EXPANSION_TRANSITION = 0.67
HIGH_VOL_RISK_OFF = 1.00

core_vol_risk_score =
  sum(probability[state] * severity[state])

Meta blend:

Default:
final_risk_score =
  0.75 * core_vol_risk_score
+ 0.25 * geometry_stress_score

This ensures geometry can modify the result but cannot fully veto vol-market risk.

Add config:
core_vol_weight: 0.75
geometry_weight: 0.25

Allow testing:
- 0.85 / 0.15
- 0.75 / 0.25
- 0.65 / 0.35

Final regime mapping:
final_risk_score:
- 0.00 to 0.25 => STABLE_LOW_VOL
- 0.25 to 0.50 => MID_VOL_CHOP
- 0.50 to 0.75 => VOL_EXPANSION_TRANSITION
- 0.75 to 1.00 => HIGH_VOL_RISK_OFF

Important safeguard:
If HMM v1 top state is VOL_EXPANSION_TRANSITION or HIGH_VOL_RISK_OFF, geometry is not allowed to downgrade final_regime by more than 1 severity level unless geometry_stress_score < 0.20 AND HMM v1 confidence < 0.55.

Example:
HMM v1 = VOL_EXPANSION_TRANSITION
geometry_stress_score = 0.25
Final can be MID_VOL_CHOP, but not STABLE_LOW_VOL.

HMM v1 = HIGH_VOL_RISK_OFF
geometry_stress_score = 0.25
Final can be VOL_EXPANSION_TRANSITION, but not MID_VOL_CHOP or STABLE_LOW_VOL.

This prevents geometry from acting like a hard veto.

Output schema:

{
  "model_name": "hmm_v3_1_meta_blend",
  "as_of_date": "...",
  "core_model": "hmm_v1_core",
  "core_vol_state": "...",
  "core_vol_probabilities": {},
  "core_vol_risk_score": 0.0,
  "geometry_stress_score": 0.0,
  "geometry_components": {
    "avg_corr_stress": 0.0,
    "eigen_stress": 0.0,
    "effective_rank_stress": 0.0,
    "log_det_stress": 0.0
  },
  "geometry_confirmation_level": "...",
  "final_risk_score": 0.0,
  "final_regime": "...",
  "confidence_adjustment": "...",
  "rationale": [],
  "warnings": []
}

Rationale examples:
- "Core vol model indicates VOL_EXPANSION_TRANSITION, but geometry stress is mild; final regime reduced to MID_VOL_CHOP."
- "Core vol model indicates elevated risk and geometry confirms one-factor stress; final regime remains VOL_EXPANSION_TRANSITION with higher confidence."
- "Geometry does not confirm stress, but downgrade capped to one severity level."

Backtest integration:
Add hmm_v3_1_meta_blend as a new model option in replay.

Run it side-by-side with:
- heuristic
- hmm_v1_core
- hmm_v2_core_plus_sector_corr
- hmm_v3_core_plus_sector_geometry
- hmm_v3_1_meta_blend

Do not remove old HMM v3.

Report additions:
Add to Model Comparison:
- hmm_v3_1_meta_blend

Add section:
"Geometry Smooth Modifier"

Show:
- core_vol_risk_score
- geometry_stress_score
- final_risk_score
- core_vol_state
- final_regime
- downgrade amount
- whether downgrade cap was applied

Disagreement Attribution:
Treat hmm_v3_1_meta_blend as separate from hmm_v3.

Key questions:
- Does v3.1 reduce false suppression vs v3?
- Does v3.1 preserve false-alarm filtering?
- Does v3.1 reduce missed risk rate?
- Does v3.1 improve risk bucket accuracy?
- Does v3.1 improve Brier Higher Vol Transition?

Tests:
1. geometry stress score returns 0-1.
2. high avg_corr/high eigen/low effective rank/low log_det produces high geometry stress.
3. low geometry stress only mildly downgrades core vol risk.
4. downgrade cap prevents VOL_EXPANSION -> STABLE_LOW_VOL unless strict exception applies.
5. final risk score respects configured weights.
6. final regime mapping works.
7. insufficient geometry history returns neutral 0.50 and warning.
8. replay can include hmm_v3_1_meta_blend without mutating HMM v1/v3.
9. report renders Geometry Smooth Modifier section.

Implementation order:
1. GeometryStressAgent.
2. core_vol_risk_score helper.
3. MetaBlendRegimeAgent.
4. hmm_v3_1_meta_blend config.
5. Replay integration.
6. Report integration.
7. Tests.
8. Run backtest over same frozen feature store/date range.

Do not implement Bayesian HMM, MCMC, RL, or automatic model promotion in this slice.