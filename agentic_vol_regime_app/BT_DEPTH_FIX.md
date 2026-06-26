Fix 10-year feature-store preflight.

VIX9D must NOT be required for the 10-year replay.

Required core series:
- SPY
- VIX
- VVIX
- VIX3M
- XLK, XLF, XLE, XLY, XLP, XLI, XLB, XLV, XLU, XLRE

Optional series:
- VIX9D
- VIX6M
- VIX9M
- VIX1Y

If VIX9D starts late:
- do not truncate the whole feature store
- set vix9d_vix_ratio to NaN before available date
- allow model configs that require VIX9D to skip/fallback
- exclude VIX9D-derived features from 10-year MLv1/HMM replay unless coverage is sufficient
- log warning, not failure

If VIX9M is missing:
- mark optional unavailable
- exclude VIX9M features
- do not fail

Sector ETFs are required for geometry.
If sector ETFs are missing from IBKR:
- verify contract type is STK
- exchange SMART
- currency USD
- use adjusted daily bars if available
- fetch each ETF individually
- report earliest available date per ETF
- do not classify sector ETFs as volatility indices

Sector ETF history is required for geometry models.

Fix IBKR sector ETF historical fetch.

Required sector ETFs:
XLK, XLF, XLE, XLY, XLP, XLI, XLB, XLV, XLU, XLRE

Fetch as:
- secType: STK
- exchange: SMART
- currency: USD
- barSize: 1 day
- whatToShow: TRADES
- useRTH: 1
- duration long enough to cover from 2010-01-01 to latest

Do not fetch sector ETFs as IND or index symbols.
Do not use CBOE exchange for sector ETFs.
Do not classify sector ETF missing history as optional.

If IBKR cannot fetch the full range in one call:
- chunk requests by year or by max IBKR duration
- stitch results
- de-duplicate by date
- sort by date
- report earliest/latest date per ETF

Required outcome:
Each sector ETF must have daily history back to at least 2013-01-01, preferably 2010-01-01.

VIX9D/VIX9M are optional.
Sector ETFs are required.

# FallBack
FEATURE_STORE_TWEAK: Add Yahoo Finance fallback for sector ETF EOD history

Problem:
IBKR only returns sector ETF history back to ~2022, which blocks 10-year geometry/rotation/MLv1 replay.

Goal:
Use IBKR as primary source, but fall back to Yahoo Finance for EOD historical sector ETF data when IBKR coverage is insufficient.

Required sector ETFs:
XLK, XLF, XLE, XLY, XLP, XLI, XLB, XLV, XLU, XLRE
Optional: XLC

Fallback source:
Yahoo Finance via yfinance or existing project data adapter.

Rules:
1. Use IBKR first.
2. If earliest IBKR date > required_start_date, fetch missing earlier EOD history from Yahoo.
3. Stitch Yahoo + IBKR.
4. Prefer IBKR where dates overlap.
5. De-duplicate by date.
6. Sort ascending.
7. Use adjusted close if available.
8. Store source provenance per row:
   - source = yahoo
   - source = ibkr
9. Do not use Yahoo intraday data.
10. Do not use Yahoo options data.
11. This fallback is EOD prices only.

Required historical coverage:
- Build sector ETF history back to 2010-01-01 if available.
- Minimum acceptable coverage for 10-year replay:
  - 2013-01-01 or earlier
- If Yahoo also fails, fail loudly with coverage_report.csv.

Create/modify:
- src/data/yahoo_eod_provider.py
- src/data/historical_source_router.py
- src/features/feature_store_builder.py
- src/backtest/hmm_replay/preflight.py

Provider behavior:
For each symbol:
- fetch IBKR history
- check earliest_date
- if earliest_date > required_start_date:
    fetch Yahoo history from required_start_date to ibkr_earliest_date
- merge histories
- validate continuous daily trading coverage

Expected output:
coverage_report.csv with:
- symbol
- required_start_date
- ibkr_earliest_date
- yahoo_earliest_date
- final_earliest_date
- final_latest_date
- row_count
- missing_days_count
- source_mix
- status

Important:
VIX9D, VIX6M, VIX9M remain optional.
Sector ETFs are required for geometry and rotation.
Do not let VIX9D truncate the full feature store.
Do not silently fall back to 3 years.

Backtest preflight should pass only if:
- SPY, VIX, VVIX, VIX3M available
- all required sector ETFs available from 2013-01-01 or earlier
- derived geometry features can be computed from 2013-01-01 onward

Tests:
1. Yahoo fallback triggers when IBKR starts too late.
2. IBKR rows override Yahoo rows on overlapping dates.
3. merged history has no duplicate dates.
4. adjusted close is used when available.
5. source provenance is preserved.
6. 10-year preflight passes with Yahoo-backed sector ETFs.
7. VIX9D missing does not fail 10-year replay. 