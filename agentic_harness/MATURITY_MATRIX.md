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

| Area | Slice | Status | Evidence / Verification | Next Upgrade Needed |
| --- | --- | --- | --- | --- |
| Runtime Ledger | SQLite runtime ledger | Production-Ready | Implemented and used as default local backend for runs, checkpoints, events, artifacts, memory, and invocations | Keep as local/dev backend; no urgent change |
| Runtime Ledger | Postgres runtime ledger | Production-Ready | Verified against local Postgres server with real `runs`, `checkpoints`, `events`, `agent_invocations`, and `memory_records` writes | Add more recovery and concurrency stress coverage |
| Runtime Ledger | JSON compatibility mirrors | Verified | Still written alongside DB-backed ledger for inspection and compatibility | Keep, but treat DB as source of truth |
| Observability | Local event recording | Production-Ready | Run events, checkpoints, and internal state are persisted and inspectable locally | Add richer telemetry querying/reporting |
| Observability | LangSmith tracing | Production-Ready | Real smoke test uploaded traces to LangSmith project `agentic_harness`; dedicated smoke scripts added | Add stronger trace naming/grouping conventions |
| Observability | LangSmith smoke scripts | Verified | `smoke_langsmith.ps1` and `smoke_langsmith.sh` added and PowerShell path executed successfully | Optionally add DAG-specific smoke script |
| Agent Runtime | First-class agent definitions | Production-Ready | Agent YAML definitions are loaded and executed through OS runtime | Expand policy surface gradually |
| Agent Runtime | Agent-bound workflow execution | Production-Ready | `run-agent` path exercised repeatedly with real agent definitions | Add richer agent contracts and output contracts |
| Workflow Runtime | Markdown workflow DSL | Production-Ready | Core example workflows execute end to end | Add more contract-driven output declarations |
| Workflow Runtime | Pause/resume at human review | Production-Ready | Verified repeatedly in `research_brief` workflow | Add stronger operator/reviewer tooling later |
| Declarative Workflows | YAML declarative workflow definitions | Verified | Definitions parse, validate, and compile into DAG plans | Broaden node types and contracts |
| Declarative Workflows | DAG compiler | Verified | Compiler builds validated execution stages and topology | Add richer compile-time policy validation |
| DAG Runtime | DAG executor | Verified | Declarative workflows execute stage by stage through OS runtime | Add more lifecycle policies around node suspension/revival |
| DAG Runtime | LangGraph-native same-stage parallelism | Verified | Replaced custom thread pool with LangGraph-native stage parallelism | Add wider concurrency and recovery stress tests |
| DAG Runtime | Human gate pause/resume | Production-Ready | Declarative DAGs pause and resume cleanly at review gates | Add richer approval policy integration |
| Memory | Ephemeral memory service | Production-Ready | Used for short-lived agent paths like `research_agent` | Add more explicit retention metrics |
| Memory | Durable/database-backed memory service | Production-Ready | Verified through `research_brief` and Postgres-backed runtime ledger | Add consolidation and promotion strategies |
| Memory | Memory retrieval/write in runtime loop | Production-Ready | Runtime retrieves and writes memory around step execution | Add better policy-driven memory classes |
| Context | Explicit context manager layer | Production-Ready | Context is assembled centrally before execution and stored in run state | Add richer budget reporting and summaries |
| Context | Rule-based compaction/budgeting | Verified | Implemented and surfaced in state and events | Add invocation-driven compaction and better heuristics |
| Tools | Toolbox service contract | Production-Ready | Tools are routed through dedicated service interface | Add more tools and tool metadata |
| Tools | `web_search` tool | Production-Ready | Real research agent runs completed successfully using live search | Add fallback provider support and tool-health reporting |
| Guardrails | Guardrail service hook | Implemented | Hook exists in runtime before/after execution | Replace pass-through default with real policy engine |
| Evaluation | Basic evaluation service | Implemented | Replaced null placeholder with basic evaluator | Add critics, scoring, and routing influence |
| Security | Security service boundary | Scaffolded | Service boundary exists in architecture/service bundle | Implement real auth, authz, and identity policy |
| Output Layer | Internal / artifact / response separation | Production-Ready | Caller-facing output is separated from internal OS state | Add more explicit typed artifact contracts |
| Output Layer | Audience-aware human vs agent output | Production-Ready | Verified with `research_agent` in both human and agent modes | Expand richer renderer/formatter options |
| LLM Integration | Optional OpenAI-backed prompt execution | Verified | Configurable LLM layer exists and is runtime pluggable | Add more providers and stronger structured output paths |
| Service Architecture | Swappable service bundle | Production-Ready | Runtime depends on service bundle, not hardcoded implementations | Add more mature alternate implementations |
| Production Tooling | Postgres smoke scripts | Verified | PowerShell and Bash smoke scripts exist and were exercised in parts | Add automated CI-friendly smoke path |
| Production Tooling | LangSmith smoke scripts | Verified | PowerShell path executed successfully; Bash path added | Run Bash path in WSL and record result |

## Near-Term Priorities

These are the main slices that are not yet mature enough:

1. Guardrails as a real policy engine.
2. Evaluation and critics that can influence retries, escalation, or human review.
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
