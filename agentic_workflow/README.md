# Agentic Workflow

Standalone LangGraph-based workflow runtime with:

- structured Markdown workflow definitions
- filesystem-backed durable memory
- explicit context manager layer with deterministic compaction
- resumable long-running workflow runs
- CLI execution and inspection
- optional LLM-backed prompt steps

## Run tests

```bash
python -m pytest
```

## Run the example workflow

```bash
python -m agentic_workflow run --workflow examples/workflows/research_brief.md --input examples/workflows/research_brief_input.json
```

That default command runs in deterministic mode with no LLM, so `prompt` steps return the rendered prompt text.

## Run with an LLM

Set your provider credentials first, for example `OPENAI_API_KEY` for OpenAI, then run:

```bash
python -m agentic_workflow run --workflow examples/workflows/research_brief.md --input examples/workflows/research_brief_input.json --llm-provider openai --model gpt-4o-mini
```

Configuration sources, in priority order:

- CLI flags: `--llm-provider`, `--model`, `--temperature`
- environment variables: `AGENTIC_WORKFLOW_LLM_PROVIDER`, `AGENTIC_WORKFLOW_MODEL`, `AGENTIC_WORKFLOW_TEMPERATURE`
- workflow frontmatter: `default_model`

If no provider is enabled, the runtime stays in no-LLM mode.

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
