from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_workflow import (
    inspect_run,
    load_workflow_definition,
    resume_workflow,
    start_workflow,
)


def _write_workflow(path: Path) -> None:
    path.write_text(
        """---
workflow_id: onboarding_workflow
title: Onboarding Workflow
entry_step: capture_request
memory_namespace: onboarding_memory
---

# Onboarding Workflow

## Step: capture_request
```yaml
type: collect
id: capture_request
output_key: request
next: classify_request
input_key: topic
memory:
  enabled: false
```

```prompt
{input.topic}
```

## Step: classify_request
```yaml
type: prompt
id: classify_request
output_key: classification
branches:
  - when: "outputs.request == 'incident'"
    next: incident_path
  - when: "outputs.request != 'incident'"
    next: standard_path
memory:
  enabled: true
  type: decision
  template: "Classification: {step_output}"
```

```prompt
Classify the request type for: {outputs.request}
```

## Step: standard_path
```yaml
type: prompt
id: standard_path
output_key: summary
memory:
  enabled: true
  type: artifact_ref
```

```prompt
Standard handling for {outputs.request}.
Prior memory:
{memory_summary}
```

## Step: incident_path
```yaml
type: human_review
id: incident_path
approved_next: finalize_incident
rejected_next: classify_request
memory:
  enabled: false
```

```prompt
Review the incident routing before finalizing.
```

## Step: finalize_incident
```yaml
type: prompt
id: finalize_incident
output_key: summary
memory:
  enabled: true
  type: artifact_ref
  template: "Final incident summary: {step_output}"
```

```prompt
Finalize the incident workflow for {outputs.request}.
```
""",
        encoding="utf-8",
    )


def test_load_workflow_definition_parses_structured_markdown(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    _write_workflow(workflow_path)

    definition = load_workflow_definition(workflow_path)

    assert definition.workflow_id == "onboarding_workflow"
    assert definition.entry_step == "capture_request"
    assert "classify_request" in definition.steps
    assert definition.steps["classify_request"].branches[0].next_step == "incident_path"
    assert definition.steps["incident_path"].approved_next == "finalize_incident"


def test_start_workflow_completes_and_persists_memory(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    storage_root = tmp_path / "runtime_store"
    _write_workflow(workflow_path)

    result = start_workflow(
        workflow_path,
        {"topic": "normal request"},
        storage_root=storage_root,
    )

    assert result["status"] == "completed"
    assert result["current_step"] is None
    assert "summary" in result["named_outputs"]
    memory_index = json.loads(
        (storage_root / "memory" / "memory_index.json").read_text(encoding="utf-8")
    )
    assert len(memory_index) == 2
    assert memory_index[0]["namespace"] == "onboarding_memory"


def test_resume_workflow_after_review(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    storage_root = tmp_path / "runtime_store"
    _write_workflow(workflow_path)

    first_result = start_workflow(
        workflow_path,
        {"topic": "incident"},
        storage_root=storage_root,
        run_id="incident-run",
    )

    assert first_result["status"] == "awaiting_review"
    assert first_result["pending_review"]["step_id"] == "incident_path"

    resumed = resume_workflow(
        "incident-run",
        storage_root=storage_root,
        decision="approved",
        notes="Looks good",
    )

    assert resumed["status"] == "completed"
    assert resumed["named_outputs"]["summary"].startswith("Finalize the incident workflow")
    inspected = inspect_run("incident-run", storage_root=storage_root)
    assert inspected["status"] == "completed"
    assert inspected["checkpoint_index"] >= 2


def test_memory_is_reused_across_runs(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    storage_root = tmp_path / "runtime_store"
    _write_workflow(workflow_path)

    start_workflow(workflow_path, {"topic": "normal request"}, storage_root=storage_root)
    second = start_workflow(
        workflow_path,
        {"topic": "normal request"},
        storage_root=storage_root,
        run_id="second-run",
    )

    assert second["status"] == "completed"
    assert second["memory_hits"], "Expected durable memory to be recalled on the second run."
    assert any(
        "Classification" in item["record"]["content"] or "Standard handling" in item["record"]["content"]
        for item in second["memory_hits"]
    )
