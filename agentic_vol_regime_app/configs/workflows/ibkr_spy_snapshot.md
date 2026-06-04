---
workflow_id: ibkr_spy_snapshot
title: IBKR SPY Snapshot
entry_step: capture_symbol
memory_namespace: ibkr_market_data_memory
description: >
  Fetch a live SPY underlying and option-chain snapshot from Interactive
  Brokers through the harness toolbox.
---

# IBKR SPY Snapshot

## Step: capture_symbol
```yaml
type: collect
id: capture_symbol
title: Capture Symbol
output_key: requested_symbol
next: fetch_ibkr_snapshot
input_key: symbol
memory:
  enabled: false
```

```prompt
{input.symbol}
```

## Step: fetch_ibkr_snapshot
```yaml
type: tool
id: fetch_ibkr_snapshot
title: Fetch IBKR Snapshot
output_key: ibkr_snapshot
tool_id: ibkr_data_pipeline
arguments:
  operation: fetch_market_snapshot
  symbol: "{outputs.requested_symbol}"
  host: "{input.host}"
  port: "{input.port}"
  client_id: "{input.client_id}"
  market_data_type: "{input.market_data_type}"
  exchange: "{input.exchange}"
  option_exchange: "{input.option_exchange}"
  currency: "{input.currency}"
  expiry_count: "{input.expiry_count}"
  strike_count: "{input.strike_count}"
  min_days_to_expiry: "{input.min_days_to_expiry}"
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
Fetched IBKR snapshot for {outputs.requested_symbol}.
```
