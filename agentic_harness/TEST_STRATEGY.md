# Test Strategy

This file defines the current test strategy for `agentic_harness`.

The goal is not just to have passing unit tests. The goal is to verify that the major `agentic_os` slices work:

- in isolation
- through the shared runtime
- against real infrastructure where that matters

## Test Layers

### 1. Fast Unit / Slice Tests

Use these while implementing a feature or refactoring internals.

What they cover:

- parser behavior
- service selection
- memory retrieval rules
- guardrail rules
- evaluation / critic rules
- runtime decision points like retry, escalate, and fail

Current high-value commands:

```bash
python -m pytest agentic_harness/tests/test_agentic_harness.py -k "structured_memory or semantic_memory or supports_structured_memory or supports_semantic_memory or supports_ephemeral_memory" -q --basetemp C:\tmp\agentic_harness_pytest
```

```bash
python -m pytest agentic_harness/tests/test_agentic_harness.py -k "guardrail or evaluation or critic" -q --basetemp C:\tmp\agentic_harness_pytest
```

### 2. Runtime Integration Tests

These verify that services behave correctly once wired into the workflow runtime.

What they should cover:

- pause/resume through human review
- guardrail escalation and approved resume
- evaluation retry and escalation
- memory retrieval/write in the runtime loop
- agent-bound workflow execution
- DAG execution and same-stage parallelism

Current primary command:

```bash
python -m pytest agentic_harness/tests/test_agentic_harness.py --basetemp C:\tmp\agentic_harness_pytest
```

### 3. Infrastructure Smoke Tests

These validate that the production-facing integrations still work on real services.

#### Postgres runtime ledger

```powershell
pwsh -File scripts/smoke_postgres.ps1
```

```bash
bash scripts/smoke_postgres.sh
```

Validates:

- runtime ledger writes
- checkpoints
- events
- agent invocations
- durable memory persistence

#### LangSmith observability

```powershell
pwsh -File scripts/smoke_langsmith.ps1
```

```bash
bash scripts/smoke_langsmith.sh
```

Validates:

- LangSmith tracing is active
- traces reach the configured project
- local runtime observability still works

#### Semantic memory / pgvector

```powershell
pwsh -File scripts/smoke_semantic_memory.ps1
```

```bash
bash scripts/smoke_semantic_memory.sh
```

Validates:

- semantic memory service wiring
- cross-run recall
- Postgres-backed embedding persistence
- `pgvector`-enabled retrieval path

### 4. Manual Operator Checks

These are worth doing after meaningful runtime changes, especially before calling a slice production-ready.

Check:

- `run-agent` in both `human` and `agent` audience modes
- `inspect` on a paused run
- resume after a guardrail or evaluation escalation
- Postgres tables for the affected run
- LangSmith trace for the same run

## Recommended Default Regression Set

For a serious runtime change, run these in order:

1. Targeted pytest slice for the changed area.
2. Combined policy/runtime slice if guardrails/evaluation/runtime changed.
3. Full `test_agentic_harness.py`.
4. Relevant smoke scripts:
   - Postgres if persistence/runtime changed
   - LangSmith if observability changed
   - semantic memory if memory/retrieval changed

## Gaps To Close

These are the most useful testing upgrades from here:

1. Add a dedicated structured-memory smoke script against Postgres.
2. Add workflow-level/default policy integration tests for guardrails and evaluation.
3. Add run-level evaluation tests once run-phase policies are implemented.
4. Add CI-friendly smoke modes that can run against disposable local services.
5. Add concurrency/recovery stress tests for multi-workflow execution.
