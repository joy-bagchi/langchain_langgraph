---
workflow_id: research_brief
title: Research Brief Workflow
entry_step: capture_request
memory_namespace: research_brief_memory
description: >
  Build a resumable research brief that pauses for human approval before
  finalizing the output.
default_model: gpt-4o-mini
---

# Research Brief Workflow

## Step: capture_request
```yaml
type: collect
id: capture_request
title: Capture Request
output_key: request
next: draft_outline
input_key: topic
memory:
  enabled: false
```

```prompt
{input.topic}
```

## Step: draft_outline
```yaml
type: prompt
id: draft_outline
title: Draft Outline
output_key: outline
next: review_outline
memory:
  enabled: true
  type: decision
  template: "Outline for {outputs.request}: {step_output}"
```

```prompt
Create a research outline for: {outputs.request}

Relevant prior memory:
{memory_summary}
```

## Step: review_outline
```yaml
type: human_review
id: review_outline
title: Review Outline
approved_next: finalize_brief
rejected_next: draft_outline
memory:
  enabled: false
```

```prompt
Review the draft outline and approve it when it is ready for finalization.
```

## Step: finalize_brief
```yaml
type: prompt
id: finalize_brief
title: Finalize Brief
output_key: final_brief
memory:
  enabled: true
  type: artifact_ref
  template: "Final brief for {outputs.request}: {step_output}"
```

```prompt
Produce the final research brief for {outputs.request} using this approved outline:
{outputs.outline}
```
