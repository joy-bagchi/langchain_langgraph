from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_harness.__main__ import _build_agent_input_payload, build_parser
from agentic_harness import (
    AgentDefinition,
    DeclarativeWorkflowDefinition,
    EphemeralMemoryService,
    ToolExecutionRequest,
    WorkflowDagBlueprint,
    YamlDeclarativeWorkflowDefinitionService,
    build_dag_blueprint,
    build_model_callable,
    build_platform_services,
    ContextManager,
    DefaultCognitiveService,
    load_declarative_workflow_definition,
    inspect_run,
    load_workflow_definition,
    extract_artifact,
    format_response,
    select_output,
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


def _write_research_workflow(path: Path) -> None:
    path.write_text(
        """---
workflow_id: research_agent_workflow
title: Research Agent Workflow
entry_step: capture_query
memory_namespace: research_agent_memory
---

# Research Agent Workflow

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
  max_results: 3
memory:
  enabled: false
```
""",
        encoding="utf-8",
    )


def _write_research_agent(path: Path, workflow_path: Path) -> None:
    path.write_text(
        f"""agent_id: research_agent
name: Research Agent
role: research_agent
workflow_path: {workflow_path.as_posix()}
llm_provider: none
memory_service_type: ephemeral
allowed_tools:
  - web_search
""",
        encoding="utf-8",
    )


def _write_declarative_workflow(path: Path, agent_path: Path) -> None:
    path.write_text(
        f"""workflow_id: quant_research_pipeline
title: Quant Research Pipeline
description: Declarative workflow that will later compile into a DAG.
entry_nodes:
  - gather_research
nodes:
  - id: gather_research
    kind: agent
    purpose: Research over the web.
    agent: {agent_path.name}
    depends_on: []
    input_bindings:
      query: "$workflow.query"
    artifact_contract: search_results@1.0
    execution_mode: real
    mock:
      enabled: true
      response:
        artifact_type: search_results
        version: "1.0"
        payload:
          results:
            - title: Mock result
              url: https://example.com/mock

  - id: extract_equations
    kind: mock_agent
    purpose: Extract equations from search results.
    depends_on:
      - gather_research
    input_bindings:
      search_results: "$node.gather_research.artifact"
    artifact_contract: math_equations@1.0
    execution_mode: mock
    mock:
      enabled: true
      response:
        artifact_type: math_equations
        version: "1.0"
        payload:
          equations:
            - expression: dF_t = alpha_t F_t^beta dW_t^1

  - id: review_output
    kind: human_gate
    purpose: Human validation gate.
    depends_on:
      - extract_equations
    input_bindings:
      packet: "$node.extract_equations.artifact"
    artifact_contract: review_packet@1.0
    execution_mode: auto
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


def test_default_toolbox_registers_web_search_tool() -> None:
    services = build_platform_services(memory_service_type="ephemeral")
    tool_ids = [tool.tool_id for tool in services.tools.list_tools()]
    assert "web_search" in tool_ids


def test_web_search_tool_executes_with_injected_client(tmp_path: Path) -> None:
    class FakeSearchClient:
        def search(self, *, query: str, max_results: int = 5, topic: str = "general", include_raw_content: bool = False):
            return {
                "query": query,
                "results": [
                    {"title": "Result A", "url": "https://example.com/a"},
                    {"title": "Result B", "url": "https://example.com/b"},
                ][:max_results],
                "topic": topic,
                "include_raw_content": include_raw_content,
            }

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        web_search_client=FakeSearchClient(),
    )
    response = services.tools.execute(
        ToolExecutionRequest(
            tool_id="web_search",
            arguments={"query": "LangGraph tool search", "max_results": 2},
        )
    )

    assert response.status == "succeeded"
    assert response.output["query"] == "LangGraph tool search"
    assert len(response.output["results"]) == 2


def test_web_search_tool_returns_unavailable_without_provider(tmp_path: Path, monkeypatch) -> None:
    from agentic_harness.agentic_os import tool_service as tool_service_module

    class MissingProvider:
        def __init__(self, *args, **kwargs) -> None:
            raise ValueError("TAVILY_API_KEY is required for the web_search tool.")

    monkeypatch.setattr(tool_service_module, "TavilyWebSearchClient", MissingProvider)
    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
    )
    response = services.tools.execute(
        ToolExecutionRequest(
            tool_id="web_search",
            arguments={"query": "LangGraph availability"},
        )
    )

    assert response.status == "unavailable"
    assert "reason" in response.metadata


def test_run_agent_parser_accepts_query_shortcut() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-agent",
            "--agent",
            "agents/research_agent.yaml",
            "--query",
            "What is an SABR model",
        ]
    )

    assert args.command == "run-agent"
    assert args.agent == "agents/research_agent.yaml"
    assert args.query == "What is an SABR model"
    assert args.output_mode == "response"


def test_build_agent_input_payload_merges_query_shortcut(tmp_path: Path) -> None:
    payload_path = tmp_path / "input.json"
    payload_path.write_text(json.dumps({"topic": "quant finance"}), encoding="utf-8")

    payload = _build_agent_input_payload(
        input_path=str(payload_path),
        query="What is an SABR model",
    )

    assert payload["topic"] == "quant finance"
    assert payload["query"] == "What is an SABR model"


def test_tool_step_executes_web_search_via_platform_tool_service(tmp_path: Path) -> None:
    workflow_path = tmp_path / "research_workflow.md"
    _write_research_workflow(workflow_path)

    class FakeSearchClient:
        def search(
            self,
            *,
            query: str,
            max_results: int = 5,
            topic: str = "general",
            include_raw_content: bool = False,
        ):
            return {
                "query": query,
                "results": [
                    {"title": "LangGraph Memory Patterns", "url": "https://example.com/langgraph-memory"},
                    {"title": "Durable Agent Workflows", "url": "https://example.com/durable-workflows"},
                    {"title": "Context Engineering", "url": "https://example.com/context-engineering"},
                ][:max_results],
                "topic": topic,
                "include_raw_content": include_raw_content,
            }

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        web_search_client=FakeSearchClient(),
    )
    result = start_workflow(
        workflow_path,
        {"query": "LangGraph durable memory"},
        services=services,
        memory_service_type="ephemeral",
        initial_state_overrides={"allowed_tools": ["web_search"]},
    )

    assert result["status"] == "completed"
    assert result["named_outputs"]["search_results"]["query"] == "LangGraph durable memory"
    assert len(result["named_outputs"]["search_results"]["results"]) == 3
    assert result["step_history"][-1]["metadata"]["tool_id"] == "web_search"


def test_tool_step_rejects_agent_when_tool_not_allowed(tmp_path: Path) -> None:
    workflow_path = tmp_path / "research_workflow.md"
    _write_research_workflow(workflow_path)

    class FakeSearchClient:
        def search(self, *, query: str, max_results: int = 5, topic: str = "general", include_raw_content: bool = False):
            return {"query": query, "results": []}

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        web_search_client=FakeSearchClient(),
    )
    result = start_workflow(
        workflow_path,
        {"query": "LangGraph durable memory"},
        services=services,
        memory_service_type="ephemeral",
        initial_state_overrides={"allowed_tools": []},
    )

    assert result["status"] == "failed"
    assert "not allowed to use tool 'web_search'" in result["last_error"]


def test_run_agent_workflow_executes_research_agent_with_web_search(tmp_path: Path) -> None:
    workflow_path = tmp_path / "research_workflow.md"
    agent_path = tmp_path / "research_agent.yaml"
    _write_research_workflow(workflow_path)
    _write_research_agent(agent_path, workflow_path)

    class FakeSearchClient:
        def search(
            self,
            *,
            query: str,
            max_results: int = 5,
            topic: str = "general",
            include_raw_content: bool = False,
        ):
            return {
                "query": query,
                "results": [
                    {"title": "Result A", "url": "https://example.com/a"},
                    {"title": "Result B", "url": "https://example.com/b"},
                ][:max_results],
                "topic": topic,
            }

    services = build_platform_services(
        storage_root=tmp_path / "runtime_store",
        memory_service_type="ephemeral",
        web_search_client=FakeSearchClient(),
    )
    result = run_agent_workflow(
        agent_path,
        {"query": "generic web search"},
        storage_root=tmp_path / "runtime_store",
        services=services,
    )

    assert result["status"] == "completed"
    assert result["agent"]["agent_id"] == "research_agent"
    assert result["agent"]["allowed_tools"] == ["web_search"]
    assert result["named_outputs"]["search_results"]["query"] == "generic web search"


def test_extract_artifact_returns_search_results_contract() -> None:
    result = {
        "run_id": "run-123",
        "workflow_id": "research_agent_search",
        "status": "completed",
        "agent_id": "research_agent",
        "agent_name": "Research Agent",
        "agent_role": "research_agent",
        "named_outputs": {
            "search_query": "What is an SABR model",
            "search_results": {
                "results": [
                    {"title": "SABR overview", "url": "https://example.com/sabr"}
                ]
            },
        },
    }

    artifact = extract_artifact(result)

    assert artifact.artifact_type == "search_results"
    assert artifact.payload["query"] == "What is an SABR model"
    assert artifact.payload["results"][0]["title"] == "SABR overview"


def test_format_response_returns_human_and_agent_views() -> None:
    artifact = extract_artifact(
        {
            "run_id": "run-123",
            "workflow_id": "research_agent_search",
            "status": "completed",
            "agent_id": "research_agent",
            "agent_name": "Research Agent",
            "agent_role": "research_agent",
            "named_outputs": {
                "search_query": "What is an SABR model",
                "search_results": {
                    "results": [
                        {"title": "SABR overview", "url": "https://example.com/sabr"}
                    ]
                },
            },
        }
    )

    human_response = format_response(artifact, audience="human", response_format="auto")
    agent_response = format_response(artifact, audience="agent", response_format="auto")

    assert human_response.response_format == "text"
    assert "Search results for: What is an SABR model" in human_response.content
    assert agent_response.response_format == "json"
    assert agent_response.content["artifact_type"] == "search_results"


def test_select_output_hides_internal_state_for_artifact_and_response() -> None:
    result = {
        "run_id": "run-123",
        "workflow_id": "research_agent_search",
        "status": "completed",
        "events": [{"event_type": "checkpoint"}],
        "named_outputs": {
            "search_query": "What is an SABR model",
            "search_results": {
                "results": [
                    {"title": "SABR overview", "url": "https://example.com/sabr"}
                ]
            },
        },
    }

    artifact_view = select_output(result, output_mode="artifact")
    response_view = select_output(result, output_mode="response", audience="agent")

    assert "events" not in artifact_view
    assert artifact_view["artifact_type"] == "search_results"
    assert "events" not in response_view
    assert response_view["response_format"] == "json"


def test_load_declarative_workflow_definition_parses_agent_and_mock_nodes(tmp_path: Path) -> None:
    agent_path = tmp_path / "research_agent.yaml"
    workflow_path = tmp_path / "pipeline.yaml"
    _write_research_agent(agent_path, tmp_path / "placeholder.md")
    _write_declarative_workflow(workflow_path, agent_path)

    definition = load_declarative_workflow_definition(workflow_path)

    assert isinstance(definition, DeclarativeWorkflowDefinition)
    assert definition.workflow_id == "quant_research_pipeline"
    assert definition.entry_nodes == ["gather_research"]
    assert definition.nodes["gather_research"].kind == "agent"
    assert definition.nodes["extract_equations"].kind == "mock_agent"
    assert definition.nodes["gather_research"].agent == str(agent_path.resolve())


def test_build_dag_blueprint_returns_roots_leaves_and_topological_order(tmp_path: Path) -> None:
    agent_path = tmp_path / "research_agent.yaml"
    workflow_path = tmp_path / "pipeline.yaml"
    _write_research_agent(agent_path, tmp_path / "placeholder.md")
    _write_declarative_workflow(workflow_path, agent_path)
    definition = load_declarative_workflow_definition(workflow_path)

    blueprint = build_dag_blueprint(definition)

    assert isinstance(blueprint, WorkflowDagBlueprint)
    assert blueprint.roots == ["gather_research"]
    assert blueprint.leaves == ["review_output"]
    assert blueprint.topological_order == [
        "gather_research",
        "extract_equations",
        "review_output",
    ]
    assert blueprint.adjacency["gather_research"] == ["extract_equations"]


def test_declarative_workflow_rejects_unknown_dependencies(tmp_path: Path) -> None:
    workflow_path = tmp_path / "pipeline.yaml"
    workflow_path.write_text(
        """workflow_id: bad_workflow
nodes:
  - id: orphan
    kind: mock_agent
    depends_on:
      - missing_node
    execution_mode: mock
    mock:
      response: {}
""",
        encoding="utf-8",
    )

    try:
        load_declarative_workflow_definition(workflow_path)
    except ValueError as exc:
        assert "depends on unknown node 'missing_node'" in str(exc)
    else:
        raise AssertionError("Expected unknown dependency validation to fail.")


def test_declarative_workflow_rejects_cycles(tmp_path: Path) -> None:
    workflow_path = tmp_path / "cyclic.yaml"
    workflow_path.write_text(
        """workflow_id: cyclic_workflow
nodes:
  - id: a
    kind: mock_agent
    depends_on:
      - b
    execution_mode: mock
    mock:
      response: {}
  - id: b
    kind: mock_agent
    depends_on:
      - a
    execution_mode: mock
    mock:
      response: {}
""",
        encoding="utf-8",
    )

    try:
        load_declarative_workflow_definition(workflow_path)
    except ValueError as exc:
        assert "contains a cycle" in str(exc)
    else:
        raise AssertionError("Expected cycle validation to fail.")


def test_platform_services_expose_declarative_workflow_loader(tmp_path: Path) -> None:
    agent_path = tmp_path / "research_agent.yaml"
    workflow_path = tmp_path / "pipeline.yaml"
    _write_research_agent(agent_path, tmp_path / "placeholder.md")
    _write_declarative_workflow(workflow_path, agent_path)

    services = build_platform_services(storage_root=tmp_path / "runtime_store")
    definition = services.declarative_workflow_definitions.load(workflow_path)

    assert isinstance(services.declarative_workflow_definitions, YamlDeclarativeWorkflowDefinitionService)
    assert definition.workflow_id == "quant_research_pipeline"

