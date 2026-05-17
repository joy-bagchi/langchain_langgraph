# Agentic Harness

Standalone LangGraph-based workflow runtime with:

- structured Markdown workflow definitions
- declarative YAML business workflow definitions for future DAG compilation
- filesystem-backed durable memory
- explicit context manager layer with deterministic compaction
- resumable long-running workflow runs
- CLI execution and inspection
- optional LLM-backed prompt steps
- toolbox service with a built-in `web_search` tool

## Run tests

```bash
python -m pytest
```

## Run the example workflow

```bash
python -m agentic_harness run --workflow examples/workflows/research_brief.md --input examples/workflows/research_brief_input.json
```

That default command runs in deterministic mode with no LLM, so `prompt` steps return the rendered prompt text.

## Run with an LLM

Set your provider credentials first, for example `OPENAI_API_KEY` for OpenAI, then run:

```bash
python -m agentic_harness run --workflow examples/workflows/research_brief.md --input examples/workflows/research_brief_input.json --llm-provider openai --model gpt-4o-mini
```

Configuration sources, in priority order:

- CLI flags: `--llm-provider`, `--model`, `--temperature`
- environment variables: `AGENTIC_HARNESS_LLM_PROVIDER`, `AGENTIC_HARNESS_MODEL`, `AGENTIC_HARNESS_TEMPERATURE`
- workflow frontmatter: `default_model`

If no provider is enabled, the runtime stays in no-LLM mode.

## Run an agent bound to a workflow

You can define an agent separately in YAML and bind it to a workflow:

```yaml
agent_id: research_analyst
name: Research Analyst
role: research_analyst
workflow_path: examples/workflows/research_brief.md
llm_provider: none
memory_service_type: filesystem
allowed_tools: []
```

Programmatic entrypoint:

```python
from agentic_harness import run_agent_workflow

result = run_agent_workflow(
    "agents/research_analyst.yaml",
    {"topic": "LangGraph patterns for durable agent memory"},
)
```

CLI entrypoint:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --input examples/workflows/research_agent_input.json
```

For quick ad hoc searches, you can skip the JSON file and pass the query directly:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model"
```

If you want to hide internal harness state and only emit the public artifact:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model" --output-mode artifact
```

If the caller is a human and you want a plain-text response instead of the full internal JSON:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model" --output-mode response --audience human
```

If the caller is another agent and you want a machine-readable handoff:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model" --output-mode response --audience agent
```

That `research_agent` is a first-class agent definition on top of `agentic_os`. Its bound workflow uses the built-in `web_search` tool for generic web search.

## Declarative Workflow Definitions

The harness now also supports a separate declarative workflow definition layer for business workflows that will later compile into DAGs.

Example:

```bash
python -c "from agentic_harness import load_declarative_workflow_definition, build_dag_blueprint, compile_workflow_dag; definition = load_declarative_workflow_definition('workflows/sabr_research_pipeline.yaml'); blueprint = build_dag_blueprint(definition); compiled = compile_workflow_dag(definition); print(blueprint.topological_order); print(compiled.execution_stages)"
```

These YAML workflows:

- define nodes declaratively
- reference real agents or mock agents
- declare dependencies with `depends_on`
- validate as DAG-ready definitions
- compile into execution stages that the future `agentic_os` executor can schedule

You can now execute a declarative DAG workflow directly:

```bash
python -m agentic_harness run-dag --workflow workflows/sabr_research_pipeline.yaml --query "What is an SABR model"
```

That example will pause at the `review_findings` human gate.

If you want to auto-complete human gates for testing:

```bash
python -m agentic_harness run-dag --workflow workflows/sabr_research_pipeline.yaml --query "What is an SABR model" --auto-approve-gates
```

If a DAG run pauses at a human gate, you can resume it:

```bash
python -m agentic_harness resume-dag --run-id <run_id> --decision approved --notes "Looks good"
```

## Built-in Tools

The default platform service bundle now includes a `web_search` tool in the toolbox.

Programmatic usage:

```python
from agentic_harness import ToolExecutionRequest, build_platform_services

services = build_platform_services()
response = services.tools.execute(
    ToolExecutionRequest(
        tool_id="web_search",
        arguments={"query": "LangGraph long running workflows", "max_results": 3},
    )
)
```

`web_search` uses Tavily when `TAVILY_API_KEY` is configured. If Tavily or the API key is not available, the tool remains registered but returns `status="unavailable"` with a reason in metadata.

The bundled `research_agent` example is configured with:

- `allowed_tools: [web_search]`
- `memory_service_type: ephemeral`
- a workflow that captures `query` input and executes a `tool` step through the toolbox service

## Context Layer

Prompt steps now run through an explicit context manager layer before execution. That layer assembles:

- retrieved memory hits
- compacted older step history
- recent step history window
- working notes
- a short `context_brief`

Templates can reference:

- `{memory_summary}`
- `{compacted_history}`
- `{context_brief}`
- `{current_task}`
- `{context.*}` for fields from the assembled context packet

