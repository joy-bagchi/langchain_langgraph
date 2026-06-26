Build an HMMBeliefAgent for the Agentic Vol Regime App.

Goal:
Add a Hidden Markov Model backed belief engine that runs in parallel with the existing Heuristic and ML agents. Do not replace existing agents. The HMM agent should infer hidden volatility regimes, estimate regime persistence, compute expected regime duration, and produce 5d/10d/21d transition probabilities for the Daily Belief Report.

Use hmmlearn GaussianHMM initially. HMMs are defined by start probabilities, transition matrix, and emissions, and hmmlearn supports posterior state probabilities through predict_proba(). Use the transition matrix for persistence and forward transition probability calculations.

Core requirements:

1. Create HMMBeliefAgent
   Suggested file:
   src/agents/hmm_belief_agent.py

2. Inputs
   Use the existing feature snapshot/history pipeline.
   Required features:
   - spy_return_1d
   - realized_vol_5d
   - realized_vol_21d
   - vix
   - vix_z_22d
   - vvix
   - vvix_vix_ratio
   - vvix_vix_z_22d
   - vix9d_vix_ratio
   - vix_vix3m_ratio
   - term_structure_slope
   - drawdown_21d
   - trend_persistence_21d

   If a feature is missing, fail gracefully with a data-quality warning. Do not silently impute zero.

3. Model
   - Use GaussianHMM.
   - Start with n_components=4.
   - covariance_type="diag" initially.
   - n_iter=500.
   - random_state fixed for reproducibility.
   - Standardize features before training.
   - Persist scaler and trained model.
   - Add config support for n_components, covariance_type, feature list, train window, and retrain cadence.

4. Initial hidden states
   Start with 4 statistical states:
   - LOW_VOL_TREND
   - MID_VOL_CHOP
   - VOL_EXPANSION
   - HIGH_VOL_STRESS

   After training, automatically map raw HMM state IDs to regime labels by ranking state summaries:
   - low average VIX + positive trend + low realized vol -> LOW_VOL_TREND
   - medium VIX + mixed trend -> MID_VOL_CHOP
   - rising vol / flattening term structure / worsening trend -> VOL_EXPANSION
   - high VIX + high realized vol + drawdown -> HIGH_VOL_STRESS

   Persist the state-label mapping.

5. Outputs
   HMMBeliefAgent should output:

   {
     "model_name": "HMMBeliefAgent",
     "model_version": "...",
     "as_of": "...",
     "state_probabilities": {
       "LOW_VOL_TREND": 0.0,
       "MID_VOL_CHOP": 0.0,
       "VOL_EXPANSION": 0.0,
       "HIGH_VOL_STRESS": 0.0
     },
     "top_state": "...",
     "transition_matrix": [[...]],
     "expected_duration_days": {
       "LOW_VOL_TREND": 0.0,
       "MID_VOL_CHOP": 0.0,
       "VOL_EXPANSION": 0.0,
       "HIGH_VOL_STRESS": 0.0
     },
     "current_state_expected_duration_days": 0.0,
     "persistence_probabilities": {
       "current_state_5d": 0.0,
       "current_state_10d": 0.0,
       "current_state_21d": 0.0
     },
     "transition_probabilities": {
       "to_high_vol_stress_5d": 0.0,
       "to_high_vol_stress_10d": 0.0,
       "to_high_vol_stress_21d": 0.0,
       "to_vol_expansion_or_high_vol_5d": 0.0,
       "to_vol_expansion_or_high_vol_10d": 0.0,
       "to_vol_expansion_or_high_vol_21d": 0.0
     },
     "confidence": 0.0,
     "warnings": [],
     "drivers": []
   }

6. Expected duration
   For each state i:
   expected_duration_days = 1 / (1 - transition_matrix[i][i])

   Guard against division by zero or transition probabilities too close to 1.

7. N-day transition probabilities
   Use matrix powers:
   P_n = transition_matrix ** n

   From the current posterior state distribution b_t:
   b_t_plus_n = b_t @ P_n

   Compute 5d, 10d, and 21d probabilities.

8. Daily Report integration
   Add a new section:

   HMM Regime Persistence

   Include:
   - HMM top state
   - HMM state probability table
   - current-state expected duration
   - probability current state persists 5d/10d/21d
   - probability of VOL_EXPANSION or HIGH_VOL_STRESS within 5d/10d/21d
   - transition matrix table
   - warnings if model is stale, untrained, or feature data is incomplete

9. Belief reconciliation
   Do not override heuristic belief yet.
   Add a comparison panel:

   | Engine | Top Regime | Confidence | Recommended Posture |
   |---|---:|---:|---|
   | Heuristic | ... | ... | ... |
   | Linear ML | ... | ... | ... |
   | HMM | ... | ... | ... |

   Then add an Ensemble placeholder, but keep it disabled until later.

10. DTE policy integration
   Add HMM outputs as optional inputs to the existing overwrite/DTE policy.
   DTE should no longer default to 1 when HMM persistence data exists.

   Rule of thumb:
   - If current regime expected duration <= 3 days: suggest 1-3 DTE
   - If expected duration 4-7 days: suggest 3-7 DTE
   - If expected duration 8-14 days: suggest 7-14 DTE
   - If expected duration > 14 days and transition risk is low: suggest 14-21 DTE
   - If high-vol transition probability is elevated: shorten DTE or require review

11. Tests
   Add tests for:
   - feature validation
   - no lookahead in training data
   - model training produces valid probability vectors
   - transition matrix rows sum to 1
   - expected duration calculation
   - 5d/10d/21d matrix-power transition calculations
   - report renders HMM section
   - graceful fallback when insufficient history exists

12. Guardrails
   - HMM output is advisory only.
   - Do not execute trades.
   - Do not auto-promote HMM output into production ensemble weighting.
   - Do not retrain on future data during replay/backtests.
   - During historical replay, train only on data available before the replay date.

Implementation priority:
1. HMMBeliefAgent class
2. model training/loading/persistence
3. current state probability output
4. transition matrix + expected duration
5. 5d/10d/21d transition probabilities
6. Daily Report HMM section
7. DTE policy integration
8. tests

Do not implement Bayesian HMM, MCMC, RL, or automatic memory learning yet.