---
workflow_id: daily_vol_regime_report
title: Daily Volatility Regime Report
entry_step: ingest_market_data
memory_namespace: volatility_regime_app_memory
description: >
  Deterministic daily volatility-regime workflow that produces a belief report,
  predictive alert, policy recommendation, and daily markdown artifact.
---

# Daily Volatility Regime Report

## Step: ingest_market_data
```yaml
type: ingest_market_data
id: ingest_market_data
title: Ingest Market Data
output_key: observation
next: validate_data_quality
memory:
  enabled: false
```

## Step: validate_data_quality
```yaml
type: validate_data_quality
id: validate_data_quality
title: Validate Data Quality
output_key: data_quality
next: compute_features
memory:
  enabled: false
```

## Step: compute_features
```yaml
type: compute_features
id: compute_features
title: Compute Features
output_key: feature_record
next: update_belief_state
memory:
  enabled: false
```

## Step: update_belief_state
```yaml
type: update_belief_state
id: update_belief_state
title: Update Belief State
output_key: belief_state
next: estimate_transition_probabilities
memory:
  enabled: false
```

## Step: estimate_transition_probabilities
```yaml
type: estimate_transition_probabilities
id: estimate_transition_probabilities
title: Estimate Transition Probabilities
output_key: transition_probabilities
next: generate_alerts
memory:
  enabled: false
```

## Step: generate_alerts
```yaml
type: generate_alerts
id: generate_alerts
title: Generate Predictive Alerts
output_key: alert_record
next: recommend_policy
memory:
  enabled: false
```

## Step: recommend_policy
```yaml
type: recommend_policy
id: recommend_policy
title: Recommend Policy
output_key: policy_recommendation
next: critic_review
memory:
  enabled: false
```

## Step: critic_review
```yaml
type: critic_review
id: critic_review
title: Critic Review
output_key: critic_review
branches:
  - when: outputs.critic_review.requires_human_review
    next: human_review_gate
  - when: outputs.alert_record.requires_human_review
    next: human_review_gate
next: persist_artifacts
memory:
  enabled: false
```

## Step: human_review_gate
```yaml
type: human_review
id: human_review_gate
title: Human Review Gate
output_key: review_decision
approved_next: persist_artifacts
rejected_next: persist_artifacts
memory:
  enabled: false
```

```prompt
Review the current daily volatility regime assessment before the artifacts and
report are finalized.
```

## Step: persist_artifacts
```yaml
type: persist_artifacts
id: persist_artifacts
title: Persist Artifacts
output_key: artifact_bundle
next: write_memory_candidates
memory:
  enabled: false
```

## Step: write_memory_candidates
```yaml
type: write_memory_candidates
id: write_memory_candidates
title: Write Memory Candidates
output_key: memory_candidates
next: produce_daily_report
memory:
  enabled: false
```

## Step: produce_daily_report
```yaml
type: produce_daily_report
id: produce_daily_report
title: Produce Daily Report
output_key: daily_report
memory:
  enabled: false
guardrails:
  post:
    block_patterns:
      - "(?i)execute live trade"
      - "(?i)place live trade"
evaluation:
  enabled: true
  min_output_chars: 300
  required_patterns:
    - "Daily Volatility Regime Report"
    - "Current regime belief favors"
    - "Predictive Alerts"
    - "Recommended action"
```
