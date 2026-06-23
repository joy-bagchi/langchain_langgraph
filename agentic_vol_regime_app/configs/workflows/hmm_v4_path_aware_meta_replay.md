---
workflow_id: hmm_v4_path_aware_meta_replay
title: HMM v4 Path-Aware Meta Replay
entry_step: load_replay_slice
memory_namespace: volatility_regime_hmm_v4_path_aware_meta
description: >
  Experimental replay-only workflow for the HMM v4 path-aware meta learner.
  This workflow is declarative documentation for the model pipeline and keeps
  the path-aware training logic encapsulated inside agentic_vol_regime_app.
---

# HMM v4 Path-Aware Meta Replay

## Step: load_replay_slice
```yaml
type: replay_load_slice
id: load_replay_slice
title: Load Replay Slice
output_key: replay_slice
next: compute_geometry_history
memory:
  enabled: false
```

## Step: compute_geometry_history
```yaml
type: replay_compute_geometry_history
id: compute_geometry_history
title: Compute Geometry Stress History
output_key: geometry_history
next: build_path_aware_features
memory:
  enabled: false
```

## Step: build_path_aware_features
```yaml
type: replay_build_path_aware_features
id: build_path_aware_features
title: Build Path-Aware Features
output_key: path_aware_features
next: build_walk_forward_dataset
memory:
  enabled: false
```

## Step: build_walk_forward_dataset
```yaml
type: replay_build_walk_forward_dataset
id: build_walk_forward_dataset
title: Build Walk-Forward Dataset
output_key: walk_forward_dataset
next: train_meta_learner
memory:
  enabled: false
```

## Step: train_meta_learner
```yaml
type: replay_train_meta_learner
id: train_meta_learner
title: Train Meta Learner
output_key: trained_meta_model
branches:
  - when: outputs.trained_meta_model.fallback_used
    next: fallback_to_v3_1
next: infer_path_aware_state
memory:
  enabled: false
```

## Step: fallback_to_v3_1
```yaml
type: replay_fallback_model
id: fallback_to_v3_1
title: Fallback To HMM v3.1 Meta-Blend
output_key: fallback_prediction
next: infer_path_aware_state
memory:
  enabled: false
```

## Step: infer_path_aware_state
```yaml
type: replay_infer_path_aware_state
id: infer_path_aware_state
title: Infer Path-Aware State
output_key: path_aware_prediction
next: render_path_aware_report
memory:
  enabled: false
guardrails:
  post:
    block_patterns:
      - "(?i)live ibkr"
      - "(?i)production artifact"
```

## Step: render_path_aware_report
```yaml
type: replay_render_path_aware_report
id: render_path_aware_report
title: Render Path-Aware Report
output_key: path_aware_report
memory:
  enabled: false
```
