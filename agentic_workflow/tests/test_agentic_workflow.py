from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_workflow import (
    AgentDefinition,
    EphemeralMemoryService,
    build_model_callable,
    build_platform_services,
    ContextManager,
    DefaultCognitiveService,
    inspect_run,
    load_workflow_definition,
    resolve_llm_config,
    render_template,
    run_agent_workflow,
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


def _write_agent(path: Path, workflow_path: Path) -> None:
    path.write_text(
        f"""agent_id: onboarding_agent
name: Onboarding Agent
role: onboarding_specialist
workflow_path: {workflow_path.as_posix()}
llm_provider: none
memory_service_type: ephemeral
allowed_tools: []
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


def test_start_workflow_uses_configurable_model_callable(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    storage_root = tmp_path / "runtime_store"
    _write_workflow(workflow_path)

    def fake_model(prompt_text: str, step, state) -> str:
        return f"LLM::{step.step_id}::{prompt_text[:20]}"

    result = start_workflow(
        workflow_path,
        {"topic": "normal request"},
        storage_root=storage_root,
        model_callable=fake_model,
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["classification"].startswith("LLM::classify_request::")
    assert result["named_outputs"]["summary"].startswith("LLM::standard_path::")


def test_resolve_llm_config_uses_workflow_default_model(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    _write_workflow(workflow_path)
    definition = load_workflow_definition(workflow_path)
    definition.default_model = "gpt-4o-mini"

    config = resolve_llm_config(
        workflow_definition=definition,
        provider="openai",
    )

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"


def test_build_model_callable_returns_none_for_disabled_config() -> None:
    config = resolve_llm_config(provider="none")
    assert build_model_callable(config) is None


def test_context_manager_compacts_older_history(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    _write_workflow(workflow_path)
    definition = load_workflow_definition(workflow_path)
    manager = ContextManager(workflow_definition=definition, max_recent_history=2)
    step = definition.steps["standard_path"]
    state = {
        "current_step": "standard_path",
        "step_history": [
            {"step_id": "capture_request", "status": "succeeded", "output": "normal request"},
            {"step_id": "classify_request", "status": "succeeded", "output": "standard"},
            {"step_id": "prior_action", "status": "succeeded", "output": "done"},
        ],
        "memory_hits": [
            {"record": {"content": "Remember the standard process."}},
        ],
        "working_notes": ["Note A"],
    }

    snapshot = manager.build_context(step, state)

    assert "capture_request -> succeeded" in snapshot.compacted_history
    assert len(snapshot.recent_history) == 2
    assert "Remember the standard process." in snapshot.memory_summary
    assert "Working notes available." in snapshot.context_brief


def test_render_template_uses_active_context_fields() -> None:
    state = {
        "input_payload": {"topic": "incident"},
        "step_outputs": {},
        "named_outputs": {},
        "memory_hits": [],
        "working_notes": [],
        "current_step": "standard_path",
        "active_context": {
            "memory_summary": "Semantic memory",
            "compacted_history": "Earlier steps compacted",
            "context_brief": "Context packet ready",
            "current_task": "Handle incident",
            "raw_memory_hits": [],
            "recent_history": [],
            "working_notes": "",
        },
    }

    rendered = render_template(
        "Task={current_task}\nBrief={context_brief}\nMem={memory_summary}\nHist={compacted_history}",
        state,
    )

    assert "Task=Handle incident" in rendered
    assert "Brief=Context packet ready" in rendered
    assert "Mem=Semantic memory" in rendered
    assert "Hist=Earlier steps compacted" in rendered


def test_workflow_run_persists_active_context(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    storage_root = tmp_path / "runtime_store"
    _write_workflow(workflow_path)

    result = start_workflow(
        workflow_path,
        {"topic": "normal request"},
        storage_root=storage_root,
    )

    assert "active_context" in result
    assert "context_brief" in result["active_context"]


def test_platform_service_bundle_supports_ephemeral_memory(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    _write_workflow(workflow_path)

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
    )
    result = start_workflow(
        workflow_path,
        {"topic": "normal request"},
        services=services,
        memory_service_type="ephemeral",
    )

    assert result["status"] == "completed"
    assert isinstance(services.memory, EphemeralMemoryService)
    assert services.memory.descriptor.implementation_id == "ephemeral_memory_service"


def test_default_cognitive_service_descriptor_exposes_capabilities() -> None:
    cognitive = DefaultCognitiveService()
    assert cognitive.descriptor.service_name == "cognitive"
    assert "deterministic_fallback" in cognitive.descriptor.capabilities


def test_run_agent_workflow_loads_agent_and_executes_bound_workflow(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.md"
    agent_path = tmp_path / "agent.yaml"
    _write_workflow(workflow_path)
    _write_agent(agent_path, workflow_path)

    result = run_agent_workflow(
        agent_path,
        {"topic": "normal request"},
        storage_root=tmp_path / "runtime_store",
    )

    assert result["status"] == "completed"
    assert result["agent"]["agent_id"] == "onboarding_agent"
    assert result["agent_role"] == "onboarding_specialist"
    assert result["named_outputs"]["summary"]
