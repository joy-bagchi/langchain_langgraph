# LangChain + LangGraph Tutorial Syllabus (Python)

This tutorial is a Python-first, two-course path aligned to:

- LangChain Academy repo: https://github.com/langchain-ai/langchain-academy
- LangChain Python docs: https://docs.langchain.com/oss/python/langchain/overview
- LangGraph Python docs: https://docs.langchain.com/oss/python/langgraph/overview

## Academy Coverage Check

The `langchain-academy` repository outlines this module sequence:

- Module 0: Basics
- Module 1: Simple graph
- Module 2: Memory
- Module 3: Human-in-the-loop
- Module 4: Parallelization
- Module 5: Research assistant
- Module 6: Deployment

This syllabus fully covers Modules `0-6` using Python, split across two courses:

- Course 1 (Intermediate): Modules `0-3` + Python LangChain fundamentals and RAG reliability.
- Course 2 (Advanced): Modules `4-6` + advanced orchestration, long-term memory, and production operations.

---

## Course 1: Intermediate LangChain + LangGraph (8 Weeks)

### Course goals

By the end of this course, you will:

- Build reliable LangChain Python pipelines.
- Build stateful LangGraph workflows with clear control flow.
- Implement memory, retrieval, and human-in-the-loop debugging patterns.
- Deliver an intermediate capstone with tests and evaluation.

### Week-by-week mapping

- Week 1: Module 0 (Basics) mapped to Python.
- Week 2: Module 1 (Simple graph).
- Week 3: Module 2 (Memory) plus retrieval foundations.
- Week 4: Module 3 (Human-in-the-loop), streaming, breakpoints, time travel.
- Week 5: RAG hardening and grounded answering.
- Week 6: Agentic RAG control loops.
- Week 7: Evaluation, regression, and reliability drills.
- Week 8: Intermediate capstone.

### Week 1: Module 0 Foundations (Python)

#### Day 1
- Session: Python environment, model setup, message primitives.
- Exercises:
  - Configure `.venv`, `.env`, and a basic chat script.
  - Run first model call and capture raw response metadata.
- Expected outputs:
  - `pyproject.toml` or `requirements.txt`, `.env.example`.
  - `src/01_hello_chat.py`.

#### Day 2
- Session: Prompt templates and reusable prompt patterns.
- Exercises:
  - Build and parameterize two prompt templates.
  - Add basic input validation.
- Expected outputs:
  - `src/02_prompt_templates.py`.

#### Day 3
- Session: Structured outputs and parsing.
- Exercises:
  - Parse model output into JSON/Pydantic-like structure.
  - Add failure handling for parse errors.
- Expected outputs:
  - `src/03_output_parsing.py`.

#### Day 4
- Session: Chains and runnable composition.
- Exercises:
  - Compose prompt -> model -> parser.
  - Package it as a reusable function.
- Expected outputs:
  - `src/04_chains.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Build CLI-style prompt lab app.
  - Add smoke test.
- Expected outputs:
  - `src/apps/prompt_lab.py`.
  - `tests/test_week1_prompt_lab.py`.

---

### Week 2: Module 1 Simple Graph

#### Day 1
- Session: Graph state and node contracts.
- Exercises:
  - Define typed shared state.
  - Build 2-node linear graph.
- Expected outputs:
  - `src/graph/01_state_and_nodes.py`.

#### Day 2
- Session: Conditional routing and reducers.
- Exercises:
  - Add branch logic based on state.
  - Add reducer-style state updates.
- Expected outputs:
  - `src/graph/02_conditional_routing.py`.

#### Day 3
- Session: Iteration loops and termination safeguards.
- Exercises:
  - Implement revise loop with max-iteration guard.
  - Persist per-iteration diagnostics.
- Expected outputs:
  - `src/graph/03_revision_loop.py`.
  - `outputs/week4_loops.json`.

#### Day 4
- Session: Resilience and retries.
- Exercises:
  - Add retry/fallback edges.
  - Handle transient tool/model failures.
- Expected outputs:
  - `src/graph/04_resilience.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Build graph-backed baseline app.
  - Add path-level tests.
- Expected outputs:
  - `src/apps/rag_graph_baseline.py`.
  - `tests/test_week4_langgraph_paths.py`.

---

### Week 3: Module 2 Memory + Retrieval Foundations

#### Day 1
- Session: Short-term memory design.
- Exercises:
  - Implement conversation memory policy.
  - Add context-window truncation.
- Expected outputs:
  - `src/memory/short_term_memory.py`.

#### Day 2
- Session: Retrieval stack basics.
- Exercises:
  - Load docs, split text, and embed chunks.
  - Build retriever with metadata filtering.
- Expected outputs:
  - `src/05_loaders.py`, `src/06_text_splitting.py`, `src/07_embeddings_vectors.py`, `src/08_retrievers.py`.

#### Day 3
- Session: Context assembly and grounded answer.
- Exercises:
  - Build context packer with source metadata.
  - Add no-evidence fallback behavior.
- Expected outputs:
  - `src/rag/context_builder.py`.
  - `src/rag/grounded_answer.py`.

#### Day 4
- Session: Memory-aware retrieval app.
- Exercises:
  - Build assistant combining memory + retrieval.
  - Add citations and trace logs.
- Expected outputs:
  - `src/apps/rag_assistant.py`.
  - `outputs/week3_citations.md`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Add tests for retrieval and memory behavior.
- Expected outputs:
  - `tests/test_week2_retrieval.py`.
  - `tests/test_week3_rag_pipeline.py`.

---

### Week 4: Module 3 Human-in-the-Loop

#### Day 1
- Session: Streaming patterns in graph runs.
- Exercises:
  - Stream intermediate node updates.
  - Capture streamed events for debugging.
- Expected outputs:
  - `src/graph/09_streaming.py`.

#### Day 2
- Session: Breakpoints and inspectability.
- Exercises:
  - Add graph breakpoints before risky nodes.
  - Inspect and edit state before resume.
- Expected outputs:
  - `src/graph/10_breakpoints.py`.
  - `docs/week4_hitl_debugging.md`.

#### Day 3
- Session: Time travel and replay.
- Exercises:
  - Re-run from prior checkpoints.
  - Compare alternative branch outcomes.
- Expected outputs:
  - `src/graph/11_time_travel.py`.

#### Day 4
- Session: HITL app workflow.
- Exercises:
  - Build review gate for approval/reject actions.
  - Route rejected steps back for correction.
- Expected outputs:
  - `src/apps/hitl_graph.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Validate streaming/breakpoint/time-travel behavior with tests.
- Expected outputs:
  - `tests/test_week4_hitl_debugging.py`.

---

### Week 5: RAG Hardening

#### Day 1
- Session: RAG pipeline architecture and interfaces.
- Exercises:
  - Formalize ingest -> retrieve -> answer interfaces.
- Expected outputs:
  - `src/rag/pipeline.py`.
  - `docs/week3_rag_architecture.md`.

#### Day 2
- Session: Query rewrite and retrieval grading.
- Exercises:
  - Add rewrite step for ambiguous queries.
  - Score retrieval coverage and relevance.
- Expected outputs:
  - `src/graph/05_query_rewrite.py`.
  - `src/graph/06_retrieval_grader.py`.

#### Day 3
- Session: Answer critique and verification.
- Exercises:
  - Add critique pass and verification loop.
  - Reject unsupported claims.
- Expected outputs:
  - `src/graph/07_answer_critique.py`.
  - `src/graph/08_verification_loop.py`.

#### Day 4
- Session: Reliability and tool guardrails.
- Exercises:
  - Add tool routing and guardrails.
  - Handle tool errors and timeouts.
- Expected outputs:
  - `src/tools/tool_router.py`.
  - `src/tools/tool_guardrails.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Run ablation experiments and compare outputs.
- Expected outputs:
  - `outputs/week3_ablation_table.csv`.
  - `docs/week3_tuning_notes.md`.

---

### Week 6: Agentic RAG Workflows

#### Day 1
- Session: End-to-end agentic graph assembly.
- Exercises:
  - Build a control-loop graph around RAG.
- Expected outputs:
  - `src/apps/agentic_rag_graph.py`.

#### Day 2
- Session: Memory + tools graph integration.
- Exercises:
  - Route tasks between retrieval and tools.
- Expected outputs:
  - `src/apps/memory_tool_graph.py`.

#### Day 3
- Session: Reliability drills.
- Exercises:
  - Simulate parser failures and empty retrieval.
- Expected outputs:
  - `docs/week6_reliability_checklist.md`.

#### Day 4
- Session: Reliability test pass.
- Exercises:
  - Add focused regression tests for known failures.
- Expected outputs:
  - `tests/test_week5_agentic_paths.py`.
  - `tests/test_week6_reliability.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Run scenario benchmark and summarize failures.
- Expected outputs:
  - `outputs/week2_retrieval.json`.

---

### Week 7: Evaluation and QA

#### Day 1
- Session: Evaluation set and rubric design.
- Exercises:
  - Build JSONL eval set.
  - Define scoring rubric.
- Expected outputs:
  - `evals/intermediate_eval_set.jsonl`.
  - `docs/week7_eval_rubric.md`.

#### Day 2
- Session: Automated evaluation harness.
- Exercises:
  - Batch-run eval prompts.
  - Persist score artifacts.
- Expected outputs:
  - `evals/run_intermediate_eval.py`.
  - `outputs/week7_eval_results.json`.

#### Day 3
- Session: Performance and optimization.
- Exercises:
  - Collect latency/token metrics.
  - Identify optimization candidates.
- Expected outputs:
  - `outputs/week7_perf_metrics.csv`.
  - `docs/week7_optimization_plan.md`.

#### Day 4
- Session: Regression stability.
- Exercises:
  - Add regression checks for earlier failures.
- Expected outputs:
  - `tests/test_week7_regressions.py`.

#### Day 5
- Session: Week integration lab.
- Exercises:
  - Compare baseline vs optimized results.
- Expected outputs:
  - `outputs/week7_delta.md`.

---

### Week 8: Intermediate Capstone

#### Day 1
- Session: Scope, acceptance criteria, and risk register.
- Expected outputs:
  - `capstone/intermediate_requirements.md`.

#### Day 2
- Session: Build sprint 1 (core functionality).
- Expected outputs:
  - `capstone/intermediate_app.py`.

#### Day 3
- Session: Build sprint 2 (graph and control flow).
- Expected outputs:
  - `capstone/intermediate_graph.py`.

#### Day 4
- Session: Validation and hardening.
- Expected outputs:
  - `capstone/intermediate_eval_report.md`.

#### Day 5
- Session: Demo and retrospective.
- Expected outputs:
  - `capstone/intermediate_demo.md`.
  - `capstone/intermediate_next_steps.md`.

---

## Course 2: Advanced LangChain + LangGraph (8 Weeks)

### Course goals

By the end of this course, you will:

- Implement advanced LangGraph patterns from Academy Modules `4-6`.
- Build research-assistant-style multi-agent systems.
- Add long-term memory and store-backed personalization.
- Deploy and operate graphs/assistants with production controls.

### Week-by-week mapping

- Week 1: Advanced graph architecture and state contracts.
- Week 2: Multi-agent orchestration.
- Week 3: Advanced retrieval and research assistant base.
- Week 4: Module 4 (Parallelization, subgraphs, map-reduce).
- Week 5: Module 5 (Research assistant refinement, reflection loops).
- Week 6: Long-term memory and LangGraph store.
- Week 7: Module 6 (Deployment, assistants, double texting).
- Week 8: Advanced capstone and defense.

### Week 1: Architecture and State at Scale

#### Day 1
- Session: Topology tradeoffs and architecture decisions.
- Expected outputs:
  - `advanced/docs/week1_topology_tradeoffs.md`.

#### Day 2
- Session: Global and subgraph state contracts.
- Expected outputs:
  - `advanced/src/state/contracts.py`.

#### Day 3
- Session: Event composition and graph interfaces.
- Expected outputs:
  - `advanced/src/graph/event_composition.py`.

#### Day 4
- Session: Failure domains and graceful degradation.
- Expected outputs:
  - `advanced/docs/week1_failure_map.md`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/graph_skeleton.py`.

---

### Week 2: Multi-Agent Orchestration

#### Day 1
- Session: Role design and capability boundaries.
- Expected outputs:
  - `advanced/docs/week2_agent_roles.md`.

#### Day 2
- Session: Delegation handoffs.
- Expected outputs:
  - `advanced/src/agents/handoffs.py`.

#### Day 3
- Session: Coordination strategies.
- Expected outputs:
  - `advanced/src/agents/coordination.py`.

#### Day 4
- Session: Arbitration and consensus.
- Expected outputs:
  - `advanced/src/agents/arbitration.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/multi_agent_graph.py`.
  - `advanced/tests/test_week2_multi_agent.py`.

---

### Week 3: Research Assistant Retrieval Core

#### Day 1
- Session: Hybrid retrieval routing.
- Expected outputs:
  - `advanced/src/retrieval/hybrid_router.py`.

#### Day 2
- Session: Reranking and context compression.
- Expected outputs:
  - `advanced/src/retrieval/rerank_and_compress.py`.

#### Day 3
- Session: Incremental indexing.
- Expected outputs:
  - `advanced/src/retrieval/incremental_index.py`.

#### Day 4
- Session: Uncertainty and clarification routing.
- Expected outputs:
  - `advanced/src/retrieval/uncertainty.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/advanced_retrieval_agent.py`.
  - `advanced/tests/test_week3_retrieval_stress.py`.

---

### Week 4: Module 4 Parallelization

#### Day 1
- Session: Parallel branches and result joining.
- Expected outputs:
  - `advanced/src/graph/parallelization.py`.

#### Day 2
- Session: Subgraph composition and reuse.
- Expected outputs:
  - `advanced/src/graph/subgraphs.py`.

#### Day 3
- Session: Map-reduce graph pattern.
- Expected outputs:
  - `advanced/src/graph/map_reduce.py`.

#### Day 4
- Session: Planner/reflection integration with parallel workflows.
- Expected outputs:
  - `advanced/src/planning/planner.py`.
  - `advanced/src/planning/reflection.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/planning_reflection_agent.py`.
  - `advanced/tests/test_week4_parallel_subgraphs.py`.

---

### Week 5: Module 5 Research Assistant Refinement

#### Day 1
- Session: Tool registry and capability discovery.
- Expected outputs:
  - `advanced/src/tools/tool_registry.py`.

#### Day 2
- Session: Tool optimization (batching/caching/dedupe).
- Expected outputs:
  - `advanced/src/tools/tool_optimizer.py`.
  - `advanced/src/tools/tool_status_dashboard.json`.

#### Day 3
- Session: HITL review payloads and checkpoints.
- Expected outputs:
  - `advanced/src/hitl/review_payload.py`.

#### Day 4
- Session: Policy guardrails and audit reports.
- Expected outputs:
  - `advanced/src/governance/policy_guardrails.py`.
  - `advanced/src/governance/audit_report.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/governed_agent.py`.
  - `advanced/tests/test_week5_governance.py`.

---

### Week 6: Long-Term Memory and Store

#### Day 1
- Session: Memory policy and retention.
- Expected outputs:
  - `advanced/docs/week6_memory_policy.md`.

#### Day 2
- Session: Long-term memory manager.
- Expected outputs:
  - `advanced/src/memory/memory_manager.py`.

#### Day 3
- Session: LangGraph store integration.
- Expected outputs:
  - `advanced/src/memory/langgraph_store.py`.

#### Day 4
- Session: Personalization and memory drift handling.
- Expected outputs:
  - `advanced/src/memory/personalization.py`.
  - `advanced/src/memory/drift_detection.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/src/apps/personalized_governed_agent.py`.
  - `advanced/tests/test_week6_personalization.py`.

---

### Week 7: Module 6 Deployment and Operations

#### Day 1
- Session: Deployment prerequisites and SLOs.
- Expected outputs:
  - `advanced/docs/week7_slos.md`.
  - `advanced/docs/week7_readiness_checklist.md`.

#### Day 2
- Session: Deploying and connecting graphs.
- Expected outputs:
  - `advanced/src/deployment/create_react_agent.py`.
  - `advanced/src/deployment/connect_deployed_graph.py`.

#### Day 3
- Session: Assistant deployment workflow.
- Expected outputs:
  - `advanced/src/deployment/deploy_assistant.py`.

#### Day 4
- Session: Handling double texting and state reducers.
- Expected outputs:
  - `advanced/src/deployment/handle_double_texting.py`.
  - `advanced/src/ops/reliability_controls.py`.

#### Day 5
- Session: Week integration lab.
- Expected outputs:
  - `advanced/tests/test_week7_deployment_flow.py`.
  - `advanced/docs/week7_deployment_runbook.md`.

---

### Week 8: Advanced Capstone

#### Day 1
- Session: Scope advanced capstone and acceptance tests.
- Expected outputs:
  - `advanced/capstone/requirements.md`.

#### Day 2
- Session: Build sprint 1 (core assistant + retrieval + orchestration).
- Expected outputs:
  - `advanced/capstone/app_core.py`.

#### Day 3
- Session: Build sprint 2 (governance + memory + deployment path).
- Expected outputs:
  - `advanced/capstone/app_governed.py`.

#### Day 4
- Session: Full validation and risk assessment.
- Expected outputs:
  - `advanced/capstone/eval_report.md`.

#### Day 5
- Session: Final defense and roadmap.
- Expected outputs:
  - `advanced/capstone/final_presentation.md`.
  - `advanced/capstone/roadmap.md`.

---

## Suggested Session Cadence (Both Courses)

- 90 min concept session
- 120 min guided implementation
- 60 min independent exercises
- 30 min review and retrospective

## Assessment Model

- 40% weekly labs and exercise completion
- 30% eval quality (grounding, correctness, robustness)
- 20% testing and operational readiness
- 10% capstone documentation and defense

## Completion Criteria

A learner is complete when they can:

- Implement Academy-equivalent patterns from Modules `0-6` in Python.
- Build and debug stateful LangGraph applications.
- Evaluate, harden, and deploy graph-based assistants.
- Defend design choices with trace and metric evidence.
