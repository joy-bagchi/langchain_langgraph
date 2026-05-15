# Agentic Workflow

Standalone LangGraph-based workflow runtime with:

- structured Markdown workflow definitions
- filesystem-backed durable memory
- resumable long-running workflow runs
- CLI execution and inspection

## Run tests

```bash
python -m pytest
```

## Run the example workflow

```bash
python -m agentic_workflow run --workflow examples/workflows/research_brief.md --input examples/workflows/research_brief_input.json
```
