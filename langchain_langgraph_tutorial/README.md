# LangChain + LangGraph Tutorial Syllabus (JavaScript)

This repository now includes a two-course learning path based on:

- LangChain JavaScript Learn docs: https://docs.langchain.com/oss/javascript/learn
- LangChain Academy Module 0 basics notebook: https://github.com/langchain-ai/langchain-academy/blob/main/module-0/basics.ipynb

The curriculum is split into:

1. Intermediate Course (8 weeks total)
2. Advanced Course (8 weeks total)

Each week has a day-by-day syllabus with:

- Session focus
- Exercises
- Expected outputs

---

## Course 1: Intermediate LangChain + LangGraph (8 Weeks)

### Course goals

By the end of this course, you will:

- Build production-style LangChain JS pipelines.
- Build and debug stateful LangGraph workflows.
- Implement retrieval, memory, tools, and evaluation loops.
- Deliver a tested mini capstone with observability and clear interfaces.

### Weekly plan overview

- Week 1: Foundations from Module 0 mapped to JS
- Week 2: Core LangChain building blocks
- Week 3: Retrieval and RAG pipeline fundamentals
- Week 4: LangGraph essentials
- Week 5: Agentic RAG with control flow
- Week 6: Memory, tool use, and reliability
- Week 7: Evaluation and quality engineering
- Week 8: Intermediate capstone build and presentation

---

### Week 1: Foundations (Module 0 -> JavaScript)

#### Day 1
- Session: Environment setup, model clients, message primitives.
- Exercises:
  - Initialize TypeScript project with LangChain and LangGraph dependencies.
  - Configure `.env` and create a simple chat completion script.
- Expected outputs:
  - `pyproject.toml` (or `requirements.txt`), `.env.example`.
  - `src/01_hello_chat.py` runnable script.

#### Day 2
- Session: Prompt templates and structured prompting.
- Exercises:
  - Build reusable prompt templates for summarization and extraction.
  - Parameterize prompt inputs and add validation checks.
- Expected outputs:
  - `src/02_prompt_templates.py`.
  - Two validated prompt templates with sample runs.

#### Day 3
- Session: Output parsers and schema-driven responses.
- Exercises:
  - Add JSON schema based output parsing.
  - Handle parser failures with retry/fallback logic.
- Expected outputs:
  - `src/03_output_parsing.py`.
  - Parsed JSON outputs and failure logs.

#### Day 4
- Session: Chains and runnable composition.
- Exercises:
  - Compose prompt -> model -> parser pipeline.
  - Build a configurable chain function with typed inputs.
- Expected outputs:
  - `src/04_chains.py`.
  - One reusable chain utility module.

#### Day 5
- Session: Week 1 integration lab.
- Exercises:
  - Build a mini "prompt lab" CLI using week components.
  - Add simple smoke tests for each command.
- Expected outputs:
  - `src/apps/prompt_lab.py`.
  - `tests/test_week1_prompt_lab.py`.

---

### Week 2: Core LangChain Components

#### Day 1
- Session: Document loaders and preprocessing.
- Exercises:
  - Load `.txt`, `.md`, and `.pdf` documents.
  - Normalize metadata and clean noisy text.
- Expected outputs:
  - `src/05_loaders.py`.
  - Shared preprocessing utility in `src/lib/preprocess.py`.

#### Day 2
- Session: Text splitting strategies.
- Exercises:
  - Compare recursive and token-aware splitters.
  - Tune chunk size and overlap for QA tasks.
- Expected outputs:
  - `src/06_text_splitting.py`.
  - Benchmark notes for splitter quality.

#### Day 3
- Session: Embeddings and vector stores.
- Exercises:
  - Embed chunks and store vectors locally.
  - Create retrieval queries with metadata filters.
- Expected outputs:
  - `src/07_embeddings_vectors.py`.
  - Serialized local vector index.

#### Day 4
- Session: Retriever interfaces and ranking.
- Exercises:
  - Implement top-k retrieval with score thresholding.
  - Add hybrid fallback (keyword + vector).
- Expected outputs:
  - `src/08_retrievers.py`.
  - Retrieval quality examples in `outputs/week2_retrieval.json`.

#### Day 5
- Session: Week 2 integration lab.
- Exercises:
  - Build a retrieval sandbox CLI for query diagnostics.
  - Add tests for chunking and retriever behavior.
- Expected outputs:
  - `src/apps/retrieval_sandbox.py`.
  - `tests/test_week2_retrieval.py`.

---

### Week 3: RAG Pipeline Fundamentals

#### Day 1
- Session: End-to-end RAG architecture.
- Exercises:
  - Define ingest, index, retrieve, and answer stages.
  - Document data flow and state interfaces.
- Expected outputs:
  - `docs/week3_rag_architecture.md`.
  - Initial scaffold `src/rag/pipeline.py`.

#### Day 2
- Session: Context assembly and citation formatting.
- Exercises:
  - Build context packer with source tracking.
  - Add citation style and traceability metadata.
- Expected outputs:
  - `src/rag/context_builder.py`.
  - Sample cited answers in `outputs/week3_citations.md`.

#### Day 3
- Session: Hallucination mitigation basics.
- Exercises:
  - Add "answer only from context" constraints.
  - Implement fallback responses for insufficient evidence.
- Expected outputs:
  - `src/rag/grounded_answer.py`.
  - Failure-mode examples with expected behavior.

#### Day 4
- Session: RAG tuning and prompt experiments.
- Exercises:
  - Compare 3 retrieval settings and 3 prompt variants.
  - Measure helpfulness and citation correctness.
- Expected outputs:
  - `outputs/week3_ablation_table.csv`.
  - Tuning notes in `docs/week3_tuning_notes.md`.

#### Day 5
- Session: Week 3 integration lab.
- Exercises:
  - Package a robust baseline RAG assistant.
  - Add tests for citation presence and no-context fallback.
- Expected outputs:
  - `src/apps/rag_assistant.py`.
  - `tests/test_week3_rag_pipeline.py`.

---

### Week 4: LangGraph Essentials

#### Day 1
- Session: Graph state model and node contracts.
- Exercises:
  - Define typed state object and update rules.
  - Build two-node linear graph.
- Expected outputs:
  - `src/graph/01_state_and_nodes.py`.
  - State schema definition and sample run logs.

#### Day 2
- Session: Conditional edges and branching.
- Exercises:
  - Route based on confidence or intent type.
  - Add branch-level logging.
- Expected outputs:
  - `src/graph/02_conditional_routing.py`.
  - Route decision traces.

#### Day 3
- Session: Loops and iterative refinement.
- Exercises:
  - Add revise-until-threshold loop with max iterations.
  - Prevent infinite loops with hard stop condition.
- Expected outputs:
  - `src/graph/03_revision_loop.py`.
  - Iteration telemetry in `outputs/week4_loops.json`.

#### Day 4
- Session: Errors, retries, and resilience.
- Exercises:
  - Implement retry policies for transient model failures.
  - Route terminal failures to a safe fallback node.
- Expected outputs:
  - `src/graph/04_resilience.py`.
  - Error handling test cases.

#### Day 5
- Session: Week 4 integration lab.
- Exercises:
  - Convert week 3 baseline RAG into LangGraph workflow.
  - Add graph unit tests for path correctness.
- Expected outputs:
  - `src/apps/rag_graph_baseline.py`.
  - `tests/test_week4_langgraph_paths.py`.

---

### Week 5: Agentic RAG with Control Flow

#### Day 1
- Session: Query analysis and rewrite node.
- Exercises:
  - Add a query rewriting and intent classification step.
  - Store original and rewritten query in state.
- Expected outputs:
  - `src/graph/05_query_rewrite.py`.
  - Query rewrite quality samples.

#### Day 2
- Session: Retrieval grading node.
- Exercises:
  - Score retrieved chunks for relevance and coverage.
  - Route low-quality retrieval to re-query path.
- Expected outputs:
  - `src/graph/06_retrieval_grader.py`.
  - Retrieval grading report.

#### Day 3
- Session: Answer drafting and critique node.
- Exercises:
  - Draft answer from selected context.
  - Add critique pass for unsupported claims.
- Expected outputs:
  - `src/graph/07_answer_critique.py`.
  - Critique logs with correction actions.

#### Day 4
- Session: Verification and revision node.
- Exercises:
  - Verify claims against citations.
  - Revise answer if verification fails.
- Expected outputs:
  - `src/graph/08_verification_loop.py`.
  - Verified output examples.

#### Day 5
- Session: Week 5 integration lab.
- Exercises:
  - Build full agentic RAG graph with control points.
  - Add tests for failure-to-recovery graph paths.
- Expected outputs:
  - `src/apps/agentic_rag_graph.py`.
  - `tests/test_week5_agentic_paths.py`.

---

### Week 6: Memory, Tools, and Reliability

#### Day 1
- Session: Short-term conversational memory.
- Exercises:
  - Add turn history summarization for context windows.
  - Implement memory truncation policy.
- Expected outputs:
  - `src/memory/short_term_memory.py`.
  - Memory behavior examples.

#### Day 2
- Session: Tool integration patterns.
- Exercises:
  - Register two tools (calculator and web-like lookup stub).
  - Build tool selection prompt and output schema.
- Expected outputs:
  - `src/tools/tool_router.py`.
  - Tool call logs with arguments/results.

#### Day 3
- Session: Tool safety and guardrails.
- Exercises:
  - Add input validation and unsafe-action rejection.
  - Add tool timeout and error fallback behavior.
- Expected outputs:
  - `src/tools/tool_guardrails.py`.
  - Guardrail test cases.

#### Day 4
- Session: Integrate memory + tools into graph.
- Exercises:
  - Add memory-aware tool-using graph node.
  - Route to retrieval or tool path by query type.
- Expected outputs:
  - `src/apps/memory_tool_graph.py`.
  - End-to-end run transcript.

#### Day 5
- Session: Week 6 integration lab.
- Exercises:
  - Run reliability drills: timeouts, empty context, parser errors.
  - Harden error messages and user-facing fallbacks.
- Expected outputs:
  - `docs/week6_reliability_checklist.md`.
  - `tests/test_week6_reliability.py`.

---

### Week 7: Evaluation and Quality Engineering

#### Day 1
- Session: Define evaluation dataset and rubric.
- Exercises:
  - Create task categories (fact QA, reasoning, synthesis).
  - Build expected-answer and citation rubric.
- Expected outputs:
  - `evals/intermediate_eval_set.jsonl`.
  - `docs/week7_eval_rubric.md`.

#### Day 2
- Session: Automated evaluation harness.
- Exercises:
  - Implement script to run batch questions through system.
  - Capture scores and failure tags.
- Expected outputs:
  - `evals/run_intermediate_eval.py`.
  - `outputs/week7_eval_results.json`.

#### Day 3
- Session: Regression testing strategy.
- Exercises:
  - Add regression cases for past failures.
  - Wire evaluation into CI-style local script.
- Expected outputs:
  - `tests/test_week7_regressions.py`.
  - `scripts/test_with_eval.ps1` or `scripts/test_with_eval.sh`.

#### Day 4
- Session: Performance and cost profiling.
- Exercises:
  - Capture token usage and latency by stage.
  - Identify top 3 optimization opportunities.
- Expected outputs:
  - `outputs/week7_perf_metrics.csv`.
  - `docs/week7_optimization_plan.md`.

#### Day 5
- Session: Week 7 integration lab.
- Exercises:
  - Apply one quality and one cost optimization.
  - Re-run eval and compare against baseline.
- Expected outputs:
  - Before/after scorecard in `outputs/week7_delta.md`.
  - Updated stable baseline branch artifacts.

---

### Week 8: Intermediate Capstone

#### Day 1
- Session: Capstone scoping and requirements.
- Exercises:
  - Choose a domain corpus and define user stories.
  - Create acceptance criteria and failure criteria.
- Expected outputs:
  - `capstone/intermediate_requirements.md`.
  - Domain dataset folder.

#### Day 2
- Session: Capstone implementation sprint 1.
- Exercises:
  - Implement ingestion, retrieval, and baseline answer path.
  - Add citation and fallback behavior.
- Expected outputs:
  - `capstone/intermediate_app.py`.
  - Working baseline demo run.

#### Day 3
- Session: Capstone implementation sprint 2.
- Exercises:
  - Add LangGraph control flow and revision loop.
  - Add memory/tool extension if needed by domain.
- Expected outputs:
  - `capstone/intermediate_graph.py`.
  - Trace logs for complex queries.

#### Day 4
- Session: Capstone testing and hardening.
- Exercises:
  - Run evaluation suite and reliability checks.
  - Fix top-priority failures only.
- Expected outputs:
  - `capstone/intermediate_eval_report.md`.
  - Stable release candidate tag notes.

#### Day 5
- Session: Demo and retrospective.
- Exercises:
  - Present architecture, key decisions, and metrics.
  - Document known limitations and next iteration plan.
- Expected outputs:
  - `capstone/intermediate_demo.md`.
  - `capstone/intermediate_next_steps.md`.

---

## Course 2: Advanced LangChain + LangGraph (8 Weeks)

### Course goals

By the end of this course, you will:

- Design robust multi-agent systems using LangGraph.
- Implement deep evaluation and model/routing strategies.
- Build long-horizon memory and human-in-the-loop workflows.
- Ship an advanced production-grade capstone with operational controls.

### Weekly plan overview

- Week 1: Advanced architecture patterns and state design
- Week 2: Multi-agent orchestration and delegation
- Week 3: Advanced retrieval systems and adaptive indexing
- Week 4: Planning, reflection, and tool ecosystems
- Week 5: Human-in-the-loop and governance controls
- Week 6: Long-term memory and personalization
- Week 7: Reliability, observability, and production operations
- Week 8: Advanced capstone and defense

---

### Week 1: Architecture and State at Scale

#### Day 1
- Session: Advanced system design patterns for LLM applications.
- Exercises:
  - Compare pipeline, graph, and multi-agent topologies.
  - Select architecture for a high-stakes domain.
- Expected outputs:
  - `advanced/docs/week1_topology_tradeoffs.md`.
  - Chosen architecture decision record.

#### Day 2
- Session: Strongly typed state and invariants.
- Exercises:
  - Define global and subgraph state contracts.
  - Add runtime state validation guards.
- Expected outputs:
  - `advanced/src/state/contracts.py`.
  - Validation test suite.

#### Day 3
- Session: Event-driven graph composition.
- Exercises:
  - Build subgraph entry/exit conventions.
  - Add event bus style state updates.
- Expected outputs:
  - `advanced/src/graph/event_composition.py`.
  - Event trace examples.

#### Day 4
- Session: Failure domains and graceful degradation.
- Exercises:
  - Map hard vs soft failure modes by component.
  - Implement degradations with clear user messaging.
- Expected outputs:
  - `advanced/docs/week1_failure_map.md`.
  - Degradation handlers.

#### Day 5
- Session: Week 1 architecture lab.
- Exercises:
  - Build a scalable graph skeleton for advanced capstone.
  - Add initial observability hooks.
- Expected outputs:
  - `advanced/src/apps/graph_skeleton.py`.
  - Baseline traces.

---

### Week 2: Multi-Agent Orchestration

#### Day 1
- Session: Agent roles and capability boundaries.
- Exercises:
  - Define planner, researcher, critic, and synthesizer agents.
  - Specify tool and context permissions per role.
- Expected outputs:
  - `advanced/docs/week2_agent_roles.md`.
  - Role schema definitions.

#### Day 2
- Session: Delegation and handoff protocols.
- Exercises:
  - Implement agent handoff node contracts.
  - Add handoff quality checks.
- Expected outputs:
  - `advanced/src/agents/handoffs.py`.
  - Handoff test transcripts.

#### Day 3
- Session: Coordination strategies.
- Exercises:
  - Compare centralized planner vs decentralized debate.
  - Implement one strategy with explicit stop criteria.
- Expected outputs:
  - `advanced/src/agents/coordination.py`.
  - Strategy comparison notes.

#### Day 4
- Session: Conflict resolution across agents.
- Exercises:
  - Add arbitration node for conflicting claims.
  - Require evidence-backed consensus.
- Expected outputs:
  - `advanced/src/agents/arbitration.py`.
  - Conflict-resolution run logs.

#### Day 5
- Session: Week 2 integration lab.
- Exercises:
  - Build an end-to-end multi-agent graph.
  - Add path tests for role sequencing correctness.
- Expected outputs:
  - `advanced/src/apps/multi_agent_graph.py`.
  - `advanced/tests/test_week2_multi_agent.py`.

---

### Week 3: Advanced Retrieval Systems

#### Day 1
- Session: Hybrid retrieval architecture.
- Exercises:
  - Implement dense + sparse + metadata retrieval.
  - Add query-dependent retriever routing.
- Expected outputs:
  - `advanced/src/retrieval/hybrid_router.py`.
  - Retrieval route diagnostics.

#### Day 2
- Session: Reranking and context compression.
- Exercises:
  - Add reranker stage and token budget control.
  - Compress context while preserving evidence.
- Expected outputs:
  - `advanced/src/retrieval/rerank_and_compress.py`.
  - Token budget comparison report.

#### Day 3
- Session: Dynamic indexing and corpus freshness.
- Exercises:
  - Add incremental indexing workflow.
  - Track index versions and invalidation rules.
- Expected outputs:
  - `advanced/src/retrieval/incremental_index.py`.
  - Index version log format.

#### Day 4
- Session: Retrieval uncertainty modeling.
- Exercises:
  - Estimate confidence using score distributions.
  - Route low-confidence requests to clarification path.
- Expected outputs:
  - `advanced/src/retrieval/uncertainty.py`.
  - Clarification decision examples.

#### Day 5
- Session: Week 3 integration lab.
- Exercises:
  - Upgrade multi-agent system with advanced retrieval stack.
  - Run retrieval stress tests.
- Expected outputs:
  - `advanced/src/apps/advanced_retrieval_agent.py`.
  - `advanced/tests/test_week3_retrieval_stress.py`.

---

### Week 4: Planning, Reflection, and Tools

#### Day 1
- Session: Task planning and decomposition.
- Exercises:
  - Build explicit planner node with executable steps.
  - Add plan validation before execution.
- Expected outputs:
  - `advanced/src/planning/planner.py`.
  - Plan validation examples.

#### Day 2
- Session: Reflective loops and self-critique.
- Exercises:
  - Implement reflection node for quality scoring.
  - Trigger selective rework based on score thresholds.
- Expected outputs:
  - `advanced/src/planning/reflection.py`.
  - Reflection audit logs.

#### Day 3
- Session: Tool ecosystems and external dependencies.
- Exercises:
  - Add tool registry with capability discovery.
  - Implement tool health checks and backoff.
- Expected outputs:
  - `advanced/src/tools/tool_registry.py`.
  - Tool status dashboard JSON.

#### Day 4
- Session: Tool-call optimization.
- Exercises:
  - Add batching, caching, and deduplication for tool requests.
  - Measure latency/cost impact.
- Expected outputs:
  - `advanced/src/tools/tool_optimizer.py`.
  - Optimization metrics report.

#### Day 5
- Session: Week 4 integration lab.
- Exercises:
  - Integrate planning + reflection + optimized tools.
  - Validate end-to-end task success rate.
- Expected outputs:
  - `advanced/src/apps/planning_reflection_agent.py`.
  - `advanced/tests/test_week4_planning.py`.

---

### Week 5: Human-in-the-Loop and Governance

#### Day 1
- Session: HITL insertion points in graph workflows.
- Exercises:
  - Identify approval gates for risky operations.
  - Design pause/resume state transitions.
- Expected outputs:
  - `advanced/docs/week5_hitl_design.md`.
  - Pause/resume state contract.

#### Day 2
- Session: Review UX and decision records.
- Exercises:
  - Build structured review payload format.
  - Log reviewer decisions and rationale.
- Expected outputs:
  - `advanced/src/hitl/review_payload.py`.
  - Decision record schema examples.

#### Day 3
- Session: Policy and compliance guardrails.
- Exercises:
  - Implement policy checks at ingress and egress.
  - Add redaction for sensitive content.
- Expected outputs:
  - `advanced/src/governance/policy_guardrails.py`.
  - Policy violation test suite.

#### Day 4
- Session: Auditability and traceability.
- Exercises:
  - Add immutable run metadata and chain-of-actions logs.
  - Generate audit report from run traces.
- Expected outputs:
  - `advanced/src/governance/audit_report.py`.
  - Sample audit report markdown.

#### Day 5
- Session: Week 5 integration lab.
- Exercises:
  - Integrate HITL and policy checks into multi-agent workflow.
  - Simulate governance edge cases.
- Expected outputs:
  - `advanced/src/apps/governed_agent.py`.
  - `advanced/tests/test_week5_governance.py`.

---

### Week 6: Long-Term Memory and Personalization

#### Day 1
- Session: Memory taxonomy and data retention rules.
- Exercises:
  - Separate episodic, semantic, and preference memory.
  - Define TTL and retention policy per memory type.
- Expected outputs:
  - `advanced/docs/week6_memory_policy.md`.
  - Memory schema definitions.

#### Day 2
- Session: Memory writing and retrieval heuristics.
- Exercises:
  - Implement memory write filters.
  - Build relevance-based memory retrieval.
- Expected outputs:
  - `advanced/src/memory/memory_manager.py`.
  - Memory retrieval test cases.

#### Day 3
- Session: Personalization under constraints.
- Exercises:
  - Add user preference conditioning to responses.
  - Ensure policy alignment and user override controls.
- Expected outputs:
  - `advanced/src/memory/personalization.py`.
  - Preference impact examples.

#### Day 4
- Session: Memory quality and drift detection.
- Exercises:
  - Detect stale/conflicting memories.
  - Implement memory update/forget operations.
- Expected outputs:
  - `advanced/src/memory/drift_detection.py`.
  - Drift report artifacts.

#### Day 5
- Session: Week 6 integration lab.
- Exercises:
  - Add long-term memory into governed multi-agent flow.
  - Validate personalization behavior with tests.
- Expected outputs:
  - `advanced/src/apps/personalized_governed_agent.py`.
  - `advanced/tests/test_week6_personalization.py`.

---

### Week 7: Reliability, Observability, and Ops

#### Day 1
- Session: SLOs and production readiness criteria.
- Exercises:
  - Define latency, quality, and cost SLOs.
  - Create production readiness checklist.
- Expected outputs:
  - `advanced/docs/week7_slos.md`.
  - `advanced/docs/week7_readiness_checklist.md`.

#### Day 2
- Session: Deep observability and tracing.
- Exercises:
  - Instrument nodes, tools, and retrievers with consistent tags.
  - Add run-level correlation IDs.
- Expected outputs:
  - `advanced/src/ops/observability.py`.
  - Correlated trace examples.

#### Day 3
- Session: Reliability engineering.
- Exercises:
  - Add circuit breakers and rate-limit handling.
  - Implement bounded retries and dead-letter paths.
- Expected outputs:
  - `advanced/src/ops/reliability_controls.py`.
  - Reliability scenario logs.

#### Day 4
- Session: Continuous evaluation in operations.
- Exercises:
  - Add scheduled eval runs and alert thresholds.
  - Compare rolling window quality metrics.
- Expected outputs:
  - `advanced/evals/run_continuous_eval.py`.
  - Alert configuration and sample alerts.

#### Day 5
- Session: Week 7 integration lab.
- Exercises:
  - Conduct a game-day simulation across failure modes.
  - Produce incident report and remediations.
- Expected outputs:
  - `advanced/docs/week7_gameday_report.md`.
  - `advanced/docs/week7_remediation_backlog.md`.

---

### Week 8: Advanced Capstone and Defense

#### Day 1
- Session: Advanced capstone scoping.
- Exercises:
  - Define complex user journeys and governance needs.
  - Establish acceptance thresholds and evaluation plan.
- Expected outputs:
  - `advanced/capstone/requirements.md`.
  - Capstone architecture draft.

#### Day 2
- Session: Capstone build sprint 1.
- Exercises:
  - Implement multi-agent orchestration and advanced retrieval.
  - Add planning and reflection loop.
- Expected outputs:
  - `advanced/capstone/app_core.py`.
  - End-to-end execution traces.

#### Day 3
- Session: Capstone build sprint 2.
- Exercises:
  - Integrate HITL, policy guardrails, and memory.
  - Harden tooling and resilience controls.
- Expected outputs:
  - `advanced/capstone/app_governed.py`.
  - Governance and reliability logs.

#### Day 4
- Session: Capstone validation.
- Exercises:
  - Run full evaluation suite and game-day scenarios.
  - Triage and fix highest-impact gaps.
- Expected outputs:
  - `advanced/capstone/eval_report.md`.
  - Final risk register.

#### Day 5
- Session: Capstone defense.
- Exercises:
  - Present architecture, metrics, and tradeoff decisions.
  - Defend design choices and future roadmap.
- Expected outputs:
  - `advanced/capstone/final_presentation.md`.
  - `advanced/capstone/roadmap.md`.

---

## Suggested Weekly Session Cadence

Use this cadence for both courses:

- 90 minutes concept session
- 120 minutes guided build
- 60 minutes independent exercises
- 30 minutes output review and reflection

## Assessment Model

- 40% weekly labs and exercise completion
- 30% evaluation metrics quality (grounding, correctness, reliability)
- 20% testing and operational readiness
- 10% final capstone presentation and documentation quality

## Prerequisites

- JavaScript/TypeScript fundamentals
- Basic async programming and API usage
- Introductory LLM prompting knowledge

## Completion Criteria

A learner is course-complete when they can:

- Design and implement stateful LangGraph systems.
- Build reliable LangChain JS pipelines with evaluation.
- Demonstrate measurable quality and reliability improvements.
- Ship documented capstones with governance and observability.

