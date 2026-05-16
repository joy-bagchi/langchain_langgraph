# Agentic Harness

Standalone LangGraph-based workflow runtime with:

- structured Markdown workflow definitions
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

That `research_agent` is a first-class agent definition on top of `agentic_os`. Its bound workflow uses the built-in `web_search` tool for generic web search.

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

