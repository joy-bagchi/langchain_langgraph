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

# Slice 2
Build Next Slice: Deterministic Live Overwrite Policy Engine

Goal:
Upgrade the existing overwrite_candidate_scorer so it becomes a deterministic policy engine fed by:

1. HMM regime probability vector
2. IBKR live market data
3. Current LEAP exposure
4. Candidate option chain data

Important:
This is NOT a predictive model.
This is NOT an autonomous trader.
This must NOT place trades.

It is a deterministic decision-support policy engine.

Architecture:

HMM model = market-state sensor
IBKR = live data source
Candidate generator = option universe builder
Overwrite scorer = deterministic policy/ranking engine
Streamlit = human decision interface

Core workflow:

1. Run or load the latest HMM model output.
2. Extract full regime probability vector.
3. Connect to IBKR.
4. Pull live underlying price for SPY or QQQ.
5. Pull live VIX.
6. Compute daily sigma:
   daily_sigma_pct = vix / sqrt(252) / 100
   daily_sigma_points = spot * daily_sigma_pct
   target_strike = spot + 0.5 * daily_sigma_points
7. Pull option chain from IBKR.
8. Generate candidate short calls around target_strike.
9. Pull bid, ask, mid, delta, IV for each candidate.
10. Feed candidates into existing score_candidates().
11. Apply deterministic policy rules.
12. Output ranked recommendation report.

Policy rules:

Use the HMM probability vector, not just selected regime.

Regime probability keys:

* low_vol_trend
* mid_vol_chop
* vol_expansion
* crash

Hard policy:

If crash >= 0.15:
mode = NO_NEW_OVERWRITE
reject all candidates unless explicit override is enabled.

If vol_expansion >= 0.55:
mode = SELECTIVE_ONLY
min_premium = base_min_premium * 1.25
min_distance_sigma = 0.50
allowed_dte = [1]
intent = only sell if premium is clearly worth upside risk.

If mid_vol_chop >= 0.50:
mode = NORMAL_OVERWRITE
min_premium = base_min_premium
min_distance_sigma = 0.35
allowed_dte = [1, 2]
intent = normal premium harvest.

If low_vol_trend >= 0.50:
mode = LIGHT_OVERWRITE
min_premium = base_min_premium
min_distance_sigma = 0.50
allowed_dte = [1]
intent = avoid capping trend too aggressively.

If no regime dominates:
mode = UNCERTAIN_SELECTIVE
min_premium = base_min_premium * 1.15
min_distance_sigma = 0.50
allowed_dte = [1]
intent = only take unusually attractive trades.

Default base_min_premium = 1.40.

Candidate generation:

Use target_strike as anchor.
Pull calls around the target strike.

Parameters:

* dte_choices default [1]
* strikes_below_target default 4
* strikes_above_target default 8
* underlying default SPY
* option type calls only

Candidate DataFrame must contain:
strike
dte
bid
ask
mid
delta
iv

Do not silently use stale/fake data.
If IBKR fails, show clear error and stop.
Manual CSV mode should remain available.

Scoring:

Keep existing scoring logic:

premium_total = mid * leap_contracts * 100

For scenario grid:
-1.0 sigma
-0.5 sigma
0 sigma
+0.5 sigma
+1.0 sigma
+1.5 sigma
+2.0 sigma

Compute:
leap_pnl
short_call_pnl
total_pnl
leap_only_pnl
overwrite_drag

score = premium_total - upside_drag_penalty * max_upside_drag

Default upside_drag_penalty = 0.35.

Decision filters:

Reject candidate if:

* policy mode blocks overwrite
* premium < policy min_premium
* distance_sigma < policy min_distance_sigma
* spread_pct > max_spread_pct
* dte not in policy allowed_dte
* bid <= 0
* mid <= 0
* ask <= bid

Default max_spread_pct = 0.25.

Add new function:

run_live_overwrite_policy_engine(
underlying: str,
regime_engine: str,
leap_contracts: int,
leap_delta: float,
base_min_premium: float,
dte_choices: list[int],
strikes_below_target: int,
strikes_above_target: int,
host: str,
port: int,
client_id: int,
market_data_type: int,
output_dir: str,
)

Return:

* scored_candidates
* scenario_table
* decision_policy
* markdown_report
* metadata dict

Streamlit changes:

Update Overwrite Scorer tab.

Modes:

1. Live IBKR + HMM
2. Manual CSV
3. Manual JSON/debug

For Live IBKR + HMM mode, display:

* underlying
* spot
* VIX
* daily sigma points
* 0.5 sigma target strike
* HMM regime probability vector
* recommendation mode
* best accepted candidate
* top accepted candidates
* top rejected candidates with reasons
* scenario PnL table for best candidate

The UI should make the trader decision obvious:

Recommended Action:

* NO OVERWRITE
  or
* SELECTIVE OVERWRITE: best candidate is SPY ###C 1DTE at mid X.XX
  or
* NORMAL OVERWRITE: best candidate is ...
  or
* LIGHT OVERWRITE: best candidate is ...

Add explanation:
Why this candidate passed.
Why others failed.

Output files:

Continue writing:
overwrite_candidates_scored.csv
overwrite_scenario_pnl.csv
overwrite_report.md

Also write:
overwrite_live_snapshot.json

Snapshot should include:
timestamp
underlying
spot
vix
daily_sigma_points
target_strike
hmm_regime_probs
selected_regime
recommendation_mode
best_candidate
input_parameters

Testing:

Add or update tests for:

1. deterministic policy selection from HMM probabilities
2. uncertain regime fallback
3. crash gate blocks all candidates
4. vol_expansion raises premium threshold and distance threshold
5. candidate generation around target strike
6. scoring still works from DataFrame
7. live orchestration works with mocked IBKR client
8. no orders are placed anywhere

Important constraints:

Do not place trades.
Do not call IBKR order APIs.
Do not mutate account state.
Do not make the scorer train or predict.
Do not make the scorer infer regimes.
Keep it deterministic and explainable.

Design principle:
The scorer is a policy engine. It consumes HMM and IBKR inputs, applies encoded policy rules, ranks candidates, and explains the recommendation.


# Slice 3
Update Overwrite Scorer UI: Decision-First Layout with Collapsed Diagnostics

Goal:
Redesign the Streamlit Overwrite Candidate Scorer tab so the primary output is a clear trading decision, not a diagnostic cockpit.

Current issue:
The page shows too many configuration fields, tables, diagnostics, paths, HMM details, and scenario data at the same level. This makes it hard to answer the trader’s actual question:

“Should I overwrite today? If yes, what strike and DTE?”

Required UX:
Make the page decision-first.

Top-level output should clearly show:

1. Recommended Action

   * NO OVERWRITE
   * SELECTIVE OVERWRITE
   * NORMAL OVERWRITE
   * LIGHT OVERWRITE

2. If overwrite is recommended:

   * underlying
   * strike
   * DTE
   * mid premium
   * distance in sigma
   * recommendation mode
   * reason the candidate was chosen

3. If no overwrite is recommended:

   * “No overwrite recommended”
   * top 3 reasons why no trade passed policy

4. Next Best Action
   Examples:

   * “Run again with 1DTE chain.”
   * “No candidate met min distance and premium thresholds.”
   * “Policy is uncertain/selective; wait for better premium.”
   * “Crash gate is active; do not open new overwrites.”

The top of the tab should feel like a Tesla cockpit, not a 747 cockpit.

Suggested top layout:

A. Main Decision Card

Use a prominent card-like section.

Example when no trade:

Recommended Action: NO OVERWRITE

Policy Mode: UNCERTAIN_SELECTIVE

Why:

* HMM is mixed: Vol Expansion 47.7%, Mid Vol Chop 34.0%.
* Policy requires 1DTE candidates.
* IBKR returned only 3DTE candidates.
* All candidates failed policy filters.

Next Best Action:
Run again when 1DTE chain is available, or intentionally evaluate 3DTE in diagnostic mode.

Example when trade passes:

Recommended Action: SELECTIVE OVERWRITE

Best Candidate:
SPY 733C, 1DTE, mid 1.52

Why:

* Distance: 0.56 sigma, above 0.50 minimum.
* Premium: 1.52, above 1.40 minimum.
* Spread: 0.08, within 0.25 maximum.
* Best portfolio score among accepted candidates.

Next Best Action:
Consider limit sell near mid or better. This is decision support only; no trade is placed.

B. Key Inputs Row

Below decision card, show only critical inputs:

* Spot
* VIX
* Daily Sigma Points
* 0.5 Sigma Target Strike
* HMM Top Regime
* Crash Gate Status
* Accepted Count / Rejected Count

Do not show output file paths at top level.

C. Collapsed Diagnostics

Everything else should be inside collapsed expanders.

Collapsed by default:

1. “Diagnostics: HMM Regime Probability Vector”

   * HMM probability table
   * selected regime
   * as-of timestamp

2. “Diagnostics: Candidate Scoring Table”

   * accepted candidates
   * rejected candidates
   * reject reasons
   * score
   * distance sigma
   * premium
   * spread pct
   * DTE

3. “Diagnostics: Scenario PnL”

   * scenario PnL table for best candidate if one exists
   * if no accepted candidate exists, show scenario PnL for closest candidate to target strike

4. “Diagnostics: IBKR / Engine Settings”

   * IBKR host
   * port
   * client ID
   * market data type
   * output directory
   * scored CSV path
   * scenario CSV path
   * report path
   * snapshot path

5. “Advanced Policy Controls”

   * min premium
   * max spread pct
   * upside drag penalty
   * strikes below target
   * strikes above target
   * allowed candidate DTE
   * allow crash-regime overwrite checkbox, if still present

Important:
Default all diagnostic/advanced expanders to collapsed.

Only the main decision card and key inputs row should be visible by default after the run.

Implementation requirements:

1. Add a helper that produces a compact decision summary from scorer results.

Create a function such as:

build_overwrite_decision_summary(
scored_candidates: pd.DataFrame,
scenario_table: pd.DataFrame,
metadata: dict,
decision_policy: DecisionPolicy,
hmm_context: HmmContext | None,
) -> dict

Return fields:

{
"recommended_action": "NO_OVERWRITE" | "SELECTIVE_OVERWRITE" | "NORMAL_OVERWRITE" | "LIGHT_OVERWRITE",
"policy_mode": "...",
"best_candidate": {...} | None,
"headline": "...",
"reason_bullets": [...],
"next_best_action": "...",
"top_regime": "...",
"top_regime_probability": ...,
"accepted_count": ...,
"rejected_count": ...,
"crash_gate_status": "OPEN" | "ACTIVE",
"diagnostic_candidate": {...} | None
}

2. Recommended action mapping:

If no accepted candidates:
recommended_action = "NO_OVERWRITE"

If accepted candidates exist:
if recommendation_mode == "SELECTIVE_ONLY":
recommended_action = "SELECTIVE_OVERWRITE"
elif recommendation_mode == "NORMAL_OVERWRITE":
recommended_action = "NORMAL_OVERWRITE"
elif recommendation_mode == "LIGHT_OVERWRITE":
recommended_action = "LIGHT_OVERWRITE"
elif recommendation_mode == "UNCERTAIN_SELECTIVE":
recommended_action = "SELECTIVE_OVERWRITE"
else:
recommended_action = "SELECTIVE_OVERWRITE"

3. Reason bullet generation:

If no accepted candidates:
Use the most common reject reasons.
Also include regime context.
Also include data issue context if applicable.

Examples:

* “Policy allowed DTE [1], but returned candidates were DTE [3].”
* “All candidates were closer than required minimum 0.50 sigma.”
* “Premium was below minimum threshold.”
* “Crash probability exceeded threshold.”
* “Bid/ask spread exceeded maximum.”

If accepted candidate exists:
Explain why the best candidate passed:
- distance_sigma
- premium vs min premium
- DTE allowed by policy
- spread within limit
- best portfolio score
- HMM policy mode

4. Next best action generation:

If no accepted candidates because DTE mismatch:
“Run again with the policy-allowed DTE chain, or intentionally add this DTE to allowed candidates if you want to evaluate it.”

If no accepted candidates because premium too low:
“Do not overwrite unless premium improves.”

If no accepted candidates because distance too close:
“Check farther OTM strikes or wait.”

If crash gate active:
“Do not open new overwrites under current crash policy.”

If accepted candidate exists:
“Review the best candidate and scenario PnL before placing any trade manually.”

5. Streamlit layout:

Replace the current top-level layout with:

st.subheader("Overwrite Policy Decision")

render_decision_card(summary)

render_key_metric_row(summary, metadata)

with st.expander("Diagnostics: HMM Regime Probability Vector", expanded=False):
show HMM table/context

with st.expander("Diagnostics: Candidate Scoring Table", expanded=False):
show accepted and rejected candidates

with st.expander("Diagnostics: Scenario PnL", expanded=False):
show scenario PnL

with st.expander("Diagnostics: IBKR / Engine Settings", expanded=False):
show settings and output paths

with st.expander("Advanced Policy Controls", expanded=False):
show tunable policy inputs

6. Move noisy controls below the decision area.

The following should not be prominent at the top:

* output directory
* IBKR host
* IBKR port
* client ID
* market data type
* output file paths
* full markdown report
* raw HMM JSON
* full candidate table
* full scenario table

They should exist, but under collapsed diagnostics.

7. Remove or demote “Allow crash-regime overwrite”

Preferred:
Remove the checkbox from the main UI.

If kept:
Move it under “Advanced Policy Controls.”
Add warning text:
“Crash override is for research only. Production policy should not open new overwrites when crash gate is active.”

8. Main card visual style:

Use simple markdown/HTML card if needed.

Make action visually obvious.

Suggested colors:

* NO OVERWRITE: warning/red/amber
* SELECTIVE OVERWRITE: amber
* NORMAL OVERWRITE: green/blue
* LIGHT OVERWRITE: blue/neutral

Do not overdo visuals.

9. Improve wording:

Use trader-readable language.

Bad:
“Accepted: 0, Rejected: 8, UNCERTAIN_SELECTIVE.”

Good:
“No overwrite recommended. The model is mixed and the live option chain did not produce any policy-valid 1DTE candidates.”

10. Keep existing files and outputs.

Do not remove:

* overwrite_candidates_scored.csv
* overwrite_scenario_pnl.csv
* overwrite_report.md
* overwrite_live_snapshot.json

But show paths only in diagnostics.

11. Tests:

Add unit tests for build_overwrite_decision_summary():

* no accepted candidates due to DTE mismatch
* no accepted candidates due to premium below threshold
* crash gate active
* accepted selective overwrite candidate
* accepted normal overwrite candidate

12. Do not change core scoring math in this UI slice.

This is a UX/readability slice only.

Do not modify:

* HMM probability logic
* candidate scoring formula
* scenario PnL formula
* IBKR data collection
* policy thresholds

Unless required only to expose summary fields.

Definition of Done:

After running the live scorer, the visible page should answer in under 5 seconds of human reading:

1. Should I overwrite?
2. If yes, what strike and DTE?
3. Why?
4. What should I check if I want more detail?

All detailed diagnostics should remain available but collapsed by default.
