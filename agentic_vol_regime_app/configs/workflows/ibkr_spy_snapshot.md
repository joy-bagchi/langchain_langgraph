---
workflow_id: ibkr_spy_snapshot
title: IBKR Market Data Reader
entry_step: fetch_daily_data
memory_namespace: ibkr_market_data_memory
description: >
  Fetch daily data and option contract/quote data through the bounded Harness
  ibkr_data_reader actions. The requested contract is always explicit.
---

# IBKR Market Data Reader

## Step: fetch_daily_data
```yaml
type: tool
id: fetch_daily_data
title: Fetch IBKR Daily Data
output_key: daily_data
next: list_contracts
tool_id: get_symbol_daily_data
arguments:
  symbol: "{input.symbol}"
  trading_date: "{input.trading_date}"
memory:
  enabled: false
```

## Step: list_contracts
```yaml
type: tool
id: list_contracts
title: List Exact Option Contracts
output_key: option_contracts
next: fetch_selected_option
tool_id: list_option_contracts
arguments:
  symbol: "{input.symbol}"
  pagination:
    limit: 100
memory:
  enabled: false
```

## Step: fetch_selected_option
```yaml
type: tool
id: fetch_selected_option
title: Fetch Explicit Option Contract
output_key: option_data
tool_id: get_option_data
arguments:
  exact_contract_id: "{input.exact_contract_id}"
memory:
  enabled: false
```

## Step: render_summary
```yaml
type: note
id: render_summary
title: Render Summary
output_key: summary
memory:
  enabled: false
```

```prompt
Fetched daily data and option contracts for {input.symbol} on {input.trading_date}. The requested exact option contract is {input.exact_contract_id}.
```
