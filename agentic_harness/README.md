# Agentic Harness

Standalone LangGraph-based workflow runtime with:

- structured Markdown workflow definitions
- declarative YAML business workflow definitions for future DAG compilation
- database-backed runtime ledger for runs, checkpoints, events, artifacts, and memory
- compatibility JSON mirrors under `.workflow_memory/` for local inspection
- explicit context manager layer with deterministic compaction
- resumable long-running workflow runs
- CLI execution and inspection
- optional LLM-backed prompt steps
- toolbox service with a built-in `web_search` tool
- optional LangSmith tracing layered on top of the local runtime ledger

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

## Runtime Ledger

The harness now persists runtime state to a database ledger by default.

- default local backend: SQLite at `.workflow_memory/runtime_ledger.db`
- production override: set `AGENTIC_HARNESS_DB_URL` to a Postgres URL
- optional CLI override: pass `--db-url`

The JSON files under `.workflow_memory/` are still written as compatibility/debug mirrors, but the database ledger is the authoritative runtime store.

One-shot local Postgres smoke test:

```powershell
pwsh -File scripts/smoke_postgres.ps1
```

WSL / Bash equivalent:

```bash
bash scripts/smoke_postgres.sh
```

The default behavior is now direct Postgres access. If you do not set `AGENTIC_HARNESS_DB_URL`, the script uses:

```text
postgresql://postgres:postgres@localhost:5432/agentic_harness
```

If you want to point it at an existing Postgres instance explicitly:

```bash
AGENTIC_HARNESS_DB_URL="postgresql://postgres:postgres@localhost:5432/agentic_harness" bash scripts/smoke_postgres.sh
```

If you already have a Postgres container running and want to reuse it by container name:

```bash
EXISTING_CONTAINER_NAME=your-postgres-container bash scripts/smoke_postgres.sh
```

If you want the script to create or manage its own temporary Docker Postgres container, opt into that behavior explicitly:

```bash
MANAGE_CONTAINER=true bash scripts/smoke_postgres.sh
```

That script will:

- connect to Postgres directly by default
- optionally reuse or manage a Docker Postgres container
- point `agentic_harness` at Postgres
- run `research_agent`
- run the durable `research_brief` workflow
- query the runtime ledger tables back through `psql`

The smoke script summary now distinguishes between:

- Postgres/runtime-ledger success
- workflow persistence success
- external tool failures such as blocked Tavily web access

So a blocked `web_search` call does not hide the fact that the Postgres integration itself passed.

## LangSmith Observability

The harness now supports LangSmith natively without removing the local runtime ledger or CLI-facing observability.

Environment variables:

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- optional `LANGSMITH_PROJECT=agentic-harness`
- optional `LANGSMITH_ENDPOINT=...`
- optional `LANGSMITH_WORKSPACE_ID=...`

LangSmith is a runtime/platform concern, not an agent-definition concern. You do not need to add LangSmith fields to agent YAML files. Any agent or workflow run can be traced by enabling LangSmith at invocation time through environment variables or CLI flags.

CLI overrides are also available on run/resume commands:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model" --langsmith-tracing --langsmith-project agentic-harness-dev
```

Behavior:

- local events, checkpoints, JSON mirrors, and runtime-ledger persistence remain enabled
- LangSmith tracing is added as an additional sink when configured
- workflow runs, DAG runs, and nested LangGraph execution run inside a LangSmith tracing context

One-shot LangSmith smoke tests:

```powershell
pwsh -File scripts/smoke_langsmith.ps1
```

```bash
bash scripts/smoke_langsmith.sh
```

Those scripts:

- run the deterministic `research_analyst` agent
- enable LangSmith tracing for that invocation
- poll LangSmith briefly for recent runs in the target project
- print a small summary of recent run ids, names, and statuses

## Run an agent bound to a workflow

You can define an agent separately in YAML and bind it to a workflow:

```yaml
agent_id: research_analyst
name: Research Analyst
role: research_analyst
workflow_path: examples/workflows/research_brief.md
llm_provider: none
memory_service_type: semantic
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

To force a specific runtime ledger:

```bash
python -m agentic_harness run-agent --agent agents/research_agent.yaml --query "What is an SABR model" --db-url "sqlite:///C:/tmp/agentic_runtime.db"
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
- `runtime_profile: default`
- a workflow that captures `query` input and executes a `tool` step through the toolbox service

## Context Layer

Prompt steps now run through an explicit context manager layer before execution. That layer assembles:

- retrieved memory hits
- compacted older step history
- recent step history window
- working notes
- a short `context_brief`

## Guardrails

The runtime now uses a rule-based guardrail engine by default.

- pre-step guardrails can block execution before a step runs
- post-step guardrails can block output or escalate it into the existing review/pause flow
- step-level overrides live in the workflow step metadata under `guardrails:`
- agent-level defaults can be provided through extra agent YAML metadata under `guardrails:`

Example step-level policy:

```yaml
type: prompt
id: draft_message
output_key: message
guardrails:
  pre:
    max_input_chars: 8000
    blocked_tool_ids: []
  post:
    max_output_chars: 12000
    detect_secrets: true
    detect_pii: true
    escalate_patterns:
      - "internal only"
```

Current default behavior:

- outputs that look like secrets are blocked
- outputs that look like PII, such as email addresses or SSNs, are escalated for review
- an approved post-step guardrail review resumes without rerunning the underlying step

## Evaluation

The runtime now also uses a rule-based evaluation service after step execution.

- evaluation runs after guardrails
- workflow-level defaults can be provided in workflow frontmatter under `evaluation:`
- step-level overrides live in workflow step metadata under `evaluation:`
- agent-level defaults can be provided through extra agent YAML metadata under `evaluation:`
- evaluation can `allow`, `retry`, `escalate`, or `fail`
- evaluation also supports a deterministic critic layer for richer scoring against workflow context

Example workflow-level default:

```yaml
---
workflow_id: research_brief
title: Research Brief Workflow
entry_step: capture_request
memory_namespace: research_brief_memory
evaluation:
  min_output_chars: 40
  critic:
    enabled: true
    required_output_keys:
      - request
    min_score: 0.8
---
```

Example step-level policy:

```yaml
type: prompt
id: draft_message
output_key: message
max_retries: 1
evaluation:
  min_output_chars: 40
  required_patterns:
    - "approved"
  escalate_patterns:
    - "needs review"
  retry_on_empty_output: true
  critic:
    enabled: true
    required_terms:
      - "SABR"
      - "stochastic volatility"
    required_step_ids:
      - "capture_request"
    required_output_keys:
      - "request"
    min_score: 0.8
    on_below_threshold: escalate
```

Current behavior:

- very short or empty outputs can trigger retry when retries are available
- missing required patterns can escalate output into the existing review/pause flow
- banned patterns can fail the step immediately
- an approved post-step evaluation review resumes without rerunning the underlying step
- critic scoring can check for required terms, prior step coverage, and required named outputs before allowing the step to pass

## Testing

The runtime now has enough production-grade surface that testing should be layered, not ad hoc.

The current test strategy is tracked in:

- [TEST_STRATEGY.md](/C:/Users/joyba/PycharmProjects/langchain_langgraph/agentic_harness/TEST_STRATEGY.md)

That file groups tests into:

- fast slice tests
- runtime integration tests
- infrastructure smoke tests
- manual operator checks

The OS now also applies rule-based context budgeting:

- runtime profiles provide default context and memory policies
- the context service estimates token load heuristically
- memory hits and recent history are compacted when the token budget is exceeded
- compaction decisions are recorded in runtime state for inspection

Templates can reference:

- `{memory_summary}`
- `{compacted_history}`
- `{context_brief}`
- `{current_task}`
- `{context.*}` for fields from the assembled context packet

## Structured and Semantic Memory

The memory layer now supports a structured memory service in addition to the existing ephemeral, durable lexical, and semantic paths.

- select it with `memory_service_type="structured"`
- structured memory persists a first-class `structured_payload` on each memory record
- recall supports `metadata_filters` and `structured_filters`, including dotted-path filters like `customer.tier`
- structured recall stays inside the runtime ledger and does not require a vector backend

- select semantic memory with `memory_service_type="semantic"`
- on Postgres with `pgvector` available, semantic recall uses vector-backed nearest-neighbor lookup
- on local SQLite or Postgres without `pgvector`, it falls back to deterministic embedded recall stored in the runtime ledger

One-shot semantic memory smoke tests:

```powershell
pwsh -File scripts/smoke_semantic_memory.ps1
```

```bash
bash scripts/smoke_semantic_memory.sh
```

Those scripts:

- generate a temporary semantic-memory agent/workflow pair with an isolated memory namespace
- run the workflow twice against the configured runtime ledger
- verify that the second run recalls memory written by the first
- confirm the Postgres `memory_embeddings` rows exist for the test namespace

If you do not set `AGENTIC_HARNESS_DB_URL`, the semantic smoke test uses:

```text
postgresql://postgres:postgres@localhost:5432/agentic_harness
```

Programmatic example:

```python
from agentic_harness import MemoryQuery, MemoryRecord, build_platform_services

services = build_platform_services(
    storage_root=".workflow_memory",
    memory_service_type="structured",
)

services.memory.remember(
    MemoryRecord.create(
        namespace="accounts",
        memory_type="profile",
        content="Enterprise renewal risk for Acme Corp.",
        source_run_id="run-1",
        source_step_id="capture_account",
        structured_payload={
            "customer": {"name": "Acme Corp", "tier": "enterprise", "region": "emea"},
            "renewal": {"month": "2026-09", "risk": "medium"},
        },
    )
)

matches = services.memory.recall(
    MemoryQuery(
        namespace="accounts",
        text="renewal",
        structured_filters={"customer.tier": "enterprise"},
    )
)
```

This keeps structured and semantic memory inside the same service architecture as the rest of `agentic_os`: the runtime uses the selected memory service, and agents do not need custom storage code.

