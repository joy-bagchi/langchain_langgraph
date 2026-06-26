---
workflow_id: research_agent_search
title: Research Agent Search
entry_step: capture_query
memory_namespace: research_agent_memory
description: Generic web search workflow for the research agent.
---

# Research Agent Search

## Step: capture_query
```yaml
type: collect
id: capture_query
output_key: search_query
next: run_web_search
input_key: query
memory:
  enabled: false
```

```prompt
{input.query}
```

## Step: run_web_search
```yaml
type: tool
id: run_web_search
output_key: search_results
tool_id: web_search
arguments:
  query: "{outputs.search_query}"
  max_results: 5
memory:
  enabled: false
```
