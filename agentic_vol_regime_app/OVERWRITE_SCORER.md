Build Slice 3: Portfolio-Aware Overwrite Candidate Scorer

Goal:
Create the first simple version of a daily report generator that ranks candidate short-call overwrites against my current LEAP position. This is NOT an execution engine. It should not place trades. It should only score candidate strikes and output a report.

Create a new script:

scripts/overwrite_candidate_scorer.py

Core purpose:
Given:
1. Current underlying price
2. Current VIX
3. Current LEAP position assumptions
4. Candidate short call strikes/premiums
5. Optional HMM regime probabilities

Produce:
1. Candidate overwrite strikes
2. Scenario PnL table
3. Portfolio-aware score
4. Accept/reject recommendation
5. Human-readable daily report

Important:
Keep this first version intentionally simple. Use manual inputs / CSV inputs first. Do not require live IBKR integration yet unless existing project utilities already make it trivial. The goal is to get the machinery in place.

Inputs:
Support CLI arguments:

--underlying SPY or QQQ
--spot current underlying price
--vix current VIX level
--leap-contracts number of long LEAP contracts
--leap-delta delta per LEAP contract
--candidate-csv path to CSV of candidate short calls
--hmm-json optional path to HMM regime probability JSON
--output-dir output directory

Candidate CSV columns:
strike
dte
bid
ask
mid
delta
iv

If mid is missing, compute mid = (bid + ask) / 2.

Daily sigma:
Compute daily implied 1-sigma move as:

daily_sigma_pct = vix / sqrt(252) / 100
daily_sigma_points = spot * daily_sigma_pct

My current heuristic target:
target_upside_points = 0.5 * daily_sigma_points
target_strike = spot + target_upside_points

For each candidate:
Compute distance_from_spot = strike - spot
Compute distance_sigma = distance_from_spot / daily_sigma_points
Compute premium = mid

Scenario grid:
Evaluate portfolio PnL under these underlying move scenarios:

-1.0 sigma
-0.5 sigma
0 sigma
+0.5 sigma
+1.0 sigma
+1.5 sigma
+2.0 sigma

For each scenario:
scenario_spot = spot + scenario_sigma * daily_sigma_points

Approximate LEAP PnL:
leap_pnl = leap_contracts * 100 * leap_delta * (scenario_spot - spot)

Short call PnL:
Assume we sell one short call per LEAP contract unless otherwise specified.
short_contracts = leap_contracts

short_call_intrinsic_at_scenario = max(0, scenario_spot - strike)
short_call_pnl = short_contracts * 100 * (premium - short_call_intrinsic_at_scenario)

Total portfolio PnL:
total_pnl = leap_pnl + short_call_pnl

Important:
This is only a first-order approximation. Do not model LEAP gamma, vega, IV change, skew, early assignment, or liquidity beyond bid/ask spread yet.

Scoring:
Create a simple score for each candidate.

Base metrics:
premium_total = premium * short_contracts * 100
pnl_flat = total_pnl at 0 sigma
pnl_plus_0_5 = total_pnl at +0.5 sigma
pnl_plus_1 = total_pnl at +1 sigma
pnl_plus_1_5 = total_pnl at +1.5 sigma
pnl_plus_2 = total_pnl at +2 sigma

Opportunity cost / upside drag:
Compare total portfolio PnL with overwrite vs LEAP-only PnL in upside scenarios.

For each upside scenario:
leap_only_pnl = leap_contracts * 100 * leap_delta * (scenario_spot - spot)
overwrite_drag = leap_only_pnl - total_pnl

Compute:
max_upside_drag = max overwrite_drag across +0.5, +1.0, +1.5, +2.0 sigma

Score:
score = premium_total - 0.35 * max_upside_drag

This penalty coefficient should be configurable:

--upside-drag-penalty default 0.35

Decision rules:
Default minimum premium per contract:
--min-premium default 1.40

Reject if:
premium < min_premium

Reject if:
distance_sigma < 0.35

Reject if:
bid/ask spread is too wide:
spread_pct = (ask - bid) / mid
default max_spread_pct = 0.25

Add CLI:
--max-spread-pct default 0.25

Regime adjustment:
If HMM JSON is provided, use it only as a policy gate.

Expected JSON format:
{
  "asof": "...",
  "regime_probs": {
    "low_vol_trend": 0.10,
    "mid_vol_chop": 0.30,
    "vol_expansion": 0.55,
    "crash": 0.05
  },
  "selected_regime": "vol_expansion"
}

Policy gate logic:
If crash probability >= 0.15:
recommendation_mode = "NO_NEW_OVERWRITE"
All candidates should be marked rejected unless override flag is passed.

If vol_expansion probability >= 0.55:
recommendation_mode = "SELECTIVE_ONLY"
Increase min premium by 25%.
Require distance_sigma >= 0.50.

If mid_vol_chop probability >= 0.50:
recommendation_mode = "NORMAL_OVERWRITE"
Use default thresholds.

If low_vol_trend probability >= 0.50:
recommendation_mode = "LIGHT_OVERWRITE"
Require distance_sigma >= 0.50, but do not raise min premium.

If no HMM JSON:
recommendation_mode = "NO_HMM_CONTEXT"
Use default thresholds.

Add CLI:
--allow-crash-overwrite false by default

Outputs:
Write three files into output-dir:

1. overwrite_candidates_scored.csv
Flat table with one row per candidate:
underlying
spot
vix
daily_sigma_points
target_strike
strike
dte
bid
ask
mid
delta
iv
distance_from_spot
distance_sigma
premium_total
spread_pct
score
decision
reject_reasons
recommendation_mode

2. overwrite_scenario_pnl.csv
Long table:
strike
dte
premium
scenario_sigma
scenario_spot
leap_pnl
short_call_pnl
total_pnl
leap_only_pnl
overwrite_drag

3. overwrite_report.md
Human-readable report with:
- timestamp
- inputs
- HMM regime probabilities if provided
- recommendation mode
- current heuristic target strike
- top 5 accepted candidates ranked by score
- top 5 rejected candidates with reasons
- scenario PnL table for best candidate
- warning that this is decision support only, not trade execution

Report language:
Make the report practical and trader-readable.

Example recommendation:
"Best candidate: SPY 743C 1DTE at mid 1.52. Distance is 0.53 sigma. Premium meets threshold. Portfolio score is 183. Recommended only as selective overwrite."

Implementation requirements:
- Use pandas and argparse.
- Use dataclasses where helpful.
- Add clear functions:
  - parse_args()
  - load_candidates()
  - load_hmm_context()
  - compute_daily_sigma()
  - score_candidates()
  - build_scenario_table()
  - apply_decision_rules()
  - write_outputs()
  - main()
- Include helpful error messages.
- Validate required CSV columns.
- Sort accepted candidates by score descending.
- Sort rejected candidates by score descending but keep reject reasons.
- Make the script deterministic and easy to test.

Also add a sample input file:

examples/overwrite_candidates_sample.csv

with a few fake rows around a sample SPY spot.

Also add a sample command in the script docstring and/or README if there is an appropriate project README:

python scripts/overwrite_candidate_scorer.py \
  --underlying SPY \
  --spot 740.25 \
  --vix 16.8 \
  --leap-contracts 5 \
  --leap-delta 0.80 \
  --candidate-csv examples/overwrite_candidates_sample.csv \
  --hmm-json outputs/latest_hmm_regime.json \
  --output-dir outputs/overwrite_scorer

Testing:
Add simple unit tests if the repo has a test framework. At minimum test:
1. daily sigma calculation
2. mid calculation from bid/ask
3. reject premium below threshold
4. vol_expansion regime raises threshold and requires farther strike
5. scenario PnL math for one simple candidate

Do not over-engineer.
Do not connect to IBKR yet unless there is already a clean existing adapter.
Do not execute trades.
Do not optimize using ML yet.
This slice is just the first portfolio-aware candidate scorer.

Clarification:
This script is a scorer, not a model.

It must not train a model.
It must not predict regimes independently.
It consumes model outputs and market inputs, then ranks overwrite candidates.

Use roles as follows:

1. HMMv3 = primary regime/policy gate
   - Use its probability vector if an HMM JSON is provided.
   - It determines overwrite mode:
     - no overwrite
     - selective only
     - normal overwrite
     - light overwrite

2. VIX 0.5-sigma heuristic = strike anchor
   - Compute daily_sigma = VIX / sqrt(252)
   - Compute target strike = spot + 0.5 * daily_sigma_points
   - Use this to evaluate how close candidate strikes are to the trader’s current heuristic target.

3. Candidate scorer = portfolio-aware evaluator
   - It ranks actual candidate short calls using:
     - premium
     - distance from spot
     - distance in sigma units
     - bid/ask spread
     - upside drag versus LEAP-only portfolio
     - scenario PnL

4. Old heuristic model
   - Do not use it as the main regime source in this slice.
   - HMMv3 should be the primary model input.
   - If HMMv3 JSON is unavailable, run in NO_HMM_CONTEXT mode using only VIX 0.5-sigma heuristic plus candidate scoring.

Important design principle:
The scorer should answer:
“Given today’s regime context, option chain, LEAP exposure, and premium, which overwrite candidate is best?”

It should not answer:
“What regime are we in?”