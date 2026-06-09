# Maturity Matrix

This file is the current maturity ledger for `agentic_harness`.

It is intended to be updated as each feature or feature-slice moves forward.

Status meanings:

- `Scaffolded`: interface or placeholder exists, but the feature is not materially implemented.
- `Implemented`: feature exists and is wired into the runtime.
- `Verified`: feature has been exercised with tests and/or local smoke runs.
- `Production-Ready`: feature is robust enough for real use in the current single-node OS scope.

Notes:

- `Production-Ready` here means production-worthy within the current scope of `agentic_harness`, not “finished forever”.
- This matrix should be updated whenever a feature materially changes state.
- Keep the entries concrete. Avoid aspirational wording in the status column.

## Current Matrix

| Area | Slice | What It Does | Status | Evidence / Verification | Next Upgrade Needed |
| --- | --- | --- | --- | --- | --- |
| Runtime Ledger | SQLite runtime ledger | Persists runs, checkpoints, events, artifacts, memory, and invocations in the default local runtime database | Production-Ready | Implemented and used as default local backend for runs, checkpoints, events, artifacts, memory, and invocations | Keep as local/dev backend; no urgent change |
| Runtime Ledger | Postgres runtime ledger | Moves the same runtime ledger model onto a production database backend | Production-Ready | Verified against local Postgres server with real `runs`, `checkpoints`, `events`, `agent_invocations`, and `memory_records` writes | Add more recovery and concurrency stress coverage |
| Runtime Ledger | JSON compatibility mirrors | Keeps filesystem-inspectable mirrors of runtime state alongside the DB ledger | Verified | Still written alongside DB-backed ledger for inspection and compatibility | Keep, but treat DB as source of truth |
| Observability | Local event recording | Records runtime events, checkpoints, and internal state for local inspection and debugging | Production-Ready | Run events, checkpoints, and internal state are persisted and inspectable locally | Add richer telemetry querying/reporting |
| Observability | LangSmith tracing | Sends workflow and agent traces to LangSmith while preserving local observability | Production-Ready | Real smoke test uploaded traces to LangSmith project `agentic_harness`; dedicated smoke scripts added | Keep improving naming, grouping, and dashboards |
| Observability | LangSmith execution tree tracing | Emits nested LangSmith spans for workflow run, memory retrieval, context prep, step execution, prompt/tool calls, guardrails, evaluation, memory writes, checkpoints, and final outputs | Verified | Runtime now emits explicit execution-tree spans; targeted tests verify nested span creation alongside the existing local event ledger | Add DAG executor spans and richer run/response summaries |
| Observability | LangSmith smoke scripts | Provides one-command validation that LangSmith tracing is wired correctly | Verified | `smoke_langsmith.ps1` and `smoke_langsmith.sh` added and PowerShell path executed successfully | Optionally add DAG-specific smoke script |
| Agent Runtime | First-class agent definitions | Lets agents be declared in YAML with workflow, memory, tool, and runtime policy bindings | Production-Ready | Agent YAML definitions are loaded and executed through OS runtime | Expand policy surface gradually |
| Agent Runtime | Agent-bound workflow execution | Runs an agent definition as a workflow-backed OS invocation | Production-Ready | `run-agent` path exercised repeatedly with real agent definitions | Add richer agent contracts and output contracts |
| Workflow Runtime | Markdown workflow DSL | Defines executable workflows in markdown with typed step metadata | Production-Ready | Core example workflows execute end to end | Add more contract-driven output declarations |
| Workflow Runtime | Pause/resume at human review | Stops workflow execution at human review gates and resumes with a decision | Production-Ready | Verified repeatedly in `research_brief` workflow | Add stronger operator/reviewer tooling later |
| Declarative Workflows | YAML declarative workflow definitions | Defines DAG-oriented business workflows separately from imperative execution code | Verified | Definitions parse, validate, and compile into DAG plans | Broaden node types and contracts |
| Declarative Workflows | DAG compiler | Turns declarative workflow definitions into validated executable DAG plans | Verified | Compiler builds validated execution stages and topology | Add richer compile-time policy validation |
| DAG Runtime | DAG executor | Executes compiled DAG plans through the runtime and service bundle | Verified | Declarative workflows execute stage by stage through OS runtime | Add more lifecycle policies around node suspension/revival |
| DAG Runtime | LangGraph-native same-stage parallelism | Runs independent DAG nodes in the same stage concurrently using LangGraph-native orchestration | Verified | Replaced custom thread pool with LangGraph-native stage parallelism | Add wider concurrency and recovery stress tests |
| DAG Runtime | Human gate pause/resume | Allows declarative DAGs to stop and resume cleanly at approval points | Production-Ready | Declarative DAGs pause and resume cleanly at review gates | Add richer approval policy integration |
| Memory | Ephemeral memory service | Supports short-lived, process-local memory for lightweight agent runs | Production-Ready | Used for short-lived agent paths like `research_agent` | Add more explicit retention metrics |
| Memory | Durable/database-backed memory service | Persists durable lexical memory records through the runtime ledger | Production-Ready | Verified through `research_brief` and Postgres-backed runtime ledger | Add consolidation and promotion strategies |
| Memory | Structured memory service | Persists structured payloads on memory records and supports field-filtered recall | Verified | `StructuredMemoryService` added, selectable through the service bundle, with nested field-filter tests over durable records and structured payload persistence in the runtime ledger | Add dedicated workflow smoke path and optional JSONB-specific indexing for Postgres |
| Memory | Semantic memory service | Adds embedding-based recall over durable memory, with `pgvector` when available | Production-Ready | Verified with targeted tests, live Postgres `pgvector` extension (`0.8.2`), direct semantic recall smoke, and end-to-end two-run workflow recall through `research_brief` using isolated semantic namespaces | Add richer embedding providers and longer-horizon lifecycle policies |
| Memory | Memory retrieval/write in runtime loop | Injects recalled memory into step context and writes memory records during execution | Production-Ready | Runtime retrieves and writes memory around step execution | Add better policy-driven memory classes |
| Context | Explicit context manager layer | Centralizes assembly of memory, history, notes, and compacted context for each step | Production-Ready | Context is assembled centrally before execution and stored in run state | Add richer budget reporting and summaries |
| Context | Rule-based compaction/budgeting | Applies deterministic limits to context size and records compaction decisions | Verified | Implemented and surfaced in state and events | Add invocation-driven compaction and better heuristics |
| Tools | Toolbox service contract | Provides a service boundary for tool registration, authorization, and execution | Production-Ready | Tools are routed through dedicated service interface | Add more tools and tool metadata |
| Tools | `web_search` tool | Gives agents a built-in web search capability through the tool service | Production-Ready | Real research agent runs completed successfully using live search | Add fallback provider support and tool-health reporting |
| Guardrails | Rule-based guardrail engine | Applies deterministic pre/post execution policy checks, blocks secret-like output, and escalates sensitive output into the review flow | Verified | Replaced the pass-through default with `RuleBasedGuardrailService`; targeted tests verify pre-step blocking, post-step escalation, and approved resume without rerunning the underlying step | Add richer rule packs, workflow-level policy defaults, and optional pre-step escalation reviews |
| Evaluation | Critic-aware rule-based evaluation engine | Applies deterministic post-step evaluation, including critic scoring over output, prior steps, and named outputs, and can allow, retry, escalate, or fail a step | Verified | Upgraded `BasicEvaluationService` into a critic-aware action engine; targeted tests verify direct retry scoring, workflow-level defaults, critic scoring, runtime retry, and post-step escalation/resume without rerunning the step | Add richer scoring dimensions and true multi-step/run-level evaluation policies |
| Security | Security service boundary | Reserves a service seam for auth, authz, and policy enforcement | Scaffolded | Service boundary exists in architecture/service bundle | Implement real auth, authz, and identity policy |
| Output Layer | Internal / artifact / response separation | Separates raw runtime state from public artifacts and caller-facing responses | Production-Ready | Caller-facing output is separated from internal OS state | Add more explicit typed artifact contracts |
| Output Layer | Audience-aware human vs agent output | Formats outputs differently for human and agent consumers | Production-Ready | Verified with `research_agent` in both human and agent modes | Expand richer renderer/formatter options |
| LLM Integration | Optional OpenAI-backed prompt execution | Lets prompt steps use pluggable model-backed execution instead of deterministic fallback only | Verified | Configurable LLM layer exists and is runtime pluggable | Add more providers and stronger structured output paths |
| Service Architecture | Swappable service bundle | Makes the runtime depend on injected services instead of hardcoded implementations | Production-Ready | Runtime depends on service bundle, not hardcoded implementations | Add more mature alternate implementations |
| Production Tooling | Postgres smoke scripts | Provides repeatable smoke tests for runtime-ledger verification against Postgres | Verified | PowerShell and Bash smoke scripts exist and were exercised in parts | Add automated CI-friendly smoke path |
| Production Tooling | LangSmith smoke scripts | Provides repeatable smoke tests for LangSmith tracing verification | Verified | PowerShell path executed successfully; Bash path added | Run Bash path in WSL and record result |
| Production Tooling | Semantic memory smoke scripts | Provides repeatable smoke tests for cross-run semantic recall on Postgres | Verified | Added PowerShell, Bash, and Python smoke paths that validate cross-run semantic recall against Postgres | Optionally add CI-friendly semantic smoke mode with mock DB bootstrap |
| Production Tooling | Test strategy document | Defines the layered verification approach across slice tests, runtime integration, smoke tests, and manual checks | Verified | Added `TEST_STRATEGY.md` with concrete commands and current coverage gaps | Add CI mappings and structured-memory smoke coverage |

## Near-Term Priorities

These are the main slices that are not yet mature enough:

1. Richer scoring dimensions and true multi-step/run-level evaluation policies.
2. Richer guardrail policies, workflow-level defaults, and safer domain-specific rule packs.
3. Invocation lifecycle management for suspend/revive beyond explicit human gates.
4. Stronger memory lifecycle management across long-running and high-agent-count workloads.
5. Security, auth, identity, and policy enforcement.
6. Explicit inter-agent artifact contracts.

## Update Rule

When updating this file for a feature:

1. Update the existing row if the slice already exists.
2. Add a new row if the slice is genuinely new.
3. Keep `Evidence / Verification` concrete.
4. Keep `Next Upgrade Needed` small and actionable.
5. Do not mark a slice `Production-Ready` unless it has been both implemented and verified in a realistic path.
