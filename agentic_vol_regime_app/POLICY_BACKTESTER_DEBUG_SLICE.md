POLICY_BACKTESTER_DEBUG_SLICE

Problem:
User cannot understand what the policy backtester is doing, and results look economically wrong.

Goal:
Make the policy backtester auditable before optimizing it.

Add a new report section:
"Policy Mechanics Audit"

For each model/policy, show:

1. Starting assumptions:
- starting_cash
- leap_enabled
- leap_entry_premium
- leap_delta
- leap_multiplier
- short_call_multiplier
- allow_naked_short_calls
- initial_spy
- start_date
- end_date

2. First 20 daily rows:
Columns:
date
spy_close
spy_change
leap_open
leap_premium_estimate
leap_daily_pnl
leap_cumulative_pnl
short_call_open
short_call_strike
short_call_entry_premium
short_call_mtm_value
short_call_daily_pnl
short_call_cumulative_pnl
total_daily_pnl
total_cumulative_pnl
action_taken
exit_reason

3. First 20 trades:
Columns:
instrument_type
entry_date
exit_date
entry_spy
exit_spy
entry_premium
exit_premium
multiplier
dollar_pnl
exit_reason

4. Add invariant checks:
- total_pnl == leap_pnl + overwrite_pnl
- no_overwrite_baseline must have overwrite_pnl == 0
- no_overwrite_baseline must have no short-call trades
- if total_pnl != 0, total_return_pct must not be 0 unless denominator is intentionally missing
- all option PnL must use multiplier 100
- short calls cannot open when LEAP is closed unless allow_naked_short_calls=true

5. Add simple deterministic scenario tests:
Scenario A:
SPY 700 → 710
LEAP delta 0.70
Expected LEAP PnL = +700

Scenario B:
Sell call at 1.50, buy back at 0.30
Expected short-call PnL = +120

Scenario C:
Sell call at 1.50, buy back at 3.00
Expected short-call PnL = -150

Scenario D:
LEAP +700 and short call -150
Expected total = +550

6. Add "Why this model made/lost money" summary:
For each model:
- LEAP contribution
- overwrite contribution
- number of short calls sold
- number of profit exits
- number of loss exits
- number of touch exits
- average short-call PnL
- worst short-call loss
- worst LEAP drawdown

Do not tune strategy until these mechanics are validated.