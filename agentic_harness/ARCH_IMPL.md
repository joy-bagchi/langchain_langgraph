# Agentic Platform Implementation Architecture

## Purpose

This document translates [ARCH.md](C:/Users/joyba/PycharmProjects/langchain_langgraph/agentic_harness/ARCH.md:1) into an implementation-oriented package map and interface specification.

The goal is to make the architecture buildable by defining:

- package boundaries
- control-plane vs runtime responsibilities
- core contracts
- execution flow
- ownership of policies, services, and workflow definitions

---

# 1. Architectural Model

## Layers vs Planes

The architecture uses two kinds of structure:

### Layers

Layers are execution strata. They participate directly in request execution.

- `Layer 1: Cognitive Layer`
- `Layer 2: Agentic OS`
- `Layer 5: Workflow & Agent Definition Layer`

### Planes

Planes are cross-cutting control systems. They influence execution but are not themselves business workflows.

- `Layer 3: Governance Plane`
- `Layer 4: Evaluation & Quality Plane`

This distinction matters because the runtime must know:

- what executes work
- what constrains work
- what evaluates work

---

# 2. Package Layout

Recommended project structure:

```text
agentic_harness/
  cognitive/
    __init__.py
    models/
      base.py
      openai.py
      anthropic.py
      local.py
      router.py
    prompts/
      templates.py
      system_prompts.py
      structured_output.py
    chains/
      prompt_chain.py
      reasoning_chain.py
    retrieval/
      embeddings.py
      rerankers.py

  agentic_os/
    __init__.py
    orchestration/
      runtime.py
      graph_compiler.py
      scheduler.py
      checkpointing.py
      routing.py
    context/
      manager.py
      compaction.py
      summarization.py
      budgeter.py
    memory/
      manager.py
      stores.py
      recall.py
      consolidation.py
      namespaces.py
    tools/
      registry.py
      executor.py
      adapters.py
      sandbox.py
      policies.py
    guardrails/
      engine.py
      validators.py
      interventions.py
    artifacts/
      manager.py
      lineage.py
      writers.py
      contracts.py
    observability/
      tracing.py
      events.py
      audit.py
      saliency.py
      telemetry.py
    reliability/
      retries.py
      idempotency.py
      recovery.py

  governance/
    __init__.py
    identity/
      principals.py
      sessions.py
    auth/
      authentication.py
      authorization.py
      entitlements.py
    policy/
      workspace_policy.py
      tool_policy.py
      data_policy.py
      approval_policy.py
    tenancy/
      tenant_context.py
      workspace_isolation.py

  evaluation/
    __init__.py
    runtime/
      critics.py
      reflection.py
      confidence.py
      hallucination.py
    offline/
      regression.py
      benchmarks.py
      datasets.py
    metrics/
      reliability.py
      latency.py
      cost.py
      tool_efficiency.py
      context_efficiency.py
    human_feedback/
      annotation.py
      review.py
      feedback_ingest.py

  definitions/
    __init__.py
    agents/
      schemas.py
      loader.py
    workflows/
      schemas.py
      loader.py
      markdown_dsl.py
    policies/
      schemas.py
      loader.py

  api/
    cli.py
    run_agent.py
    run_workflow.py
    inspect_run.py

  shared/
    types/
    ids.py
    enums.py
    errors.py
    utils.py
```

This is the target package shape. The current codebase should evolve toward it incrementally rather than via one large rewrite.

---

# 3. Ownership Boundaries

## Cognitive Layer

Owns:

- provider/client abstraction
- prompt and reasoning primitives
- model invocation
- embeddings/reranking
- structured outputs
- token/cost metadata from providers

Does not own:

- workflow logic
- orchestration
- memory lifecycle
- business policy
- tool authorization

## Agentic OS

Owns:

- execution runtime
- workflow graph compilation
- context assembly
- memory services
- tool execution
- artifact writing
- guardrail execution
- retries/checkpointing/recovery
- runtime event emission

Does not own:

- who is allowed to do what
- enterprise access policy
- offline benchmarking policy
- business workflow definitions

## Governance Plane

Owns:

- principal identity
- authorization and entitlements
- workspace/tenant isolation
- approval policy
- tool/data/workspace access policy

Does not own:

- graph execution
- prompt assembly
- step routing

## Evaluation Plane

Owns:

- runtime critics
- offline test suites
- scoring rubrics
- quality judgments
- human feedback ingestion

Does not own:

- orchestration itself
- primary workflow logic

## Workflow & Agent Definition Layer

Owns:

- declarative workflow contracts
- declarative agent contracts
- execution policy references
- memory policy references
- tool bundle references
- artifact expectations

Does not own:

- implementation of runtime services
- model clients
- authorization engine

---

# 4. Core Domain Objects

These contracts should exist early and remain stable.

## 4.1 Cognitive Contracts

```python
class ModelRequest:
    prompt: str
    system_prompt: str | None
    model: str
    temperature: float
    response_format: dict | None
    metadata: dict

class ModelResponse:
    content: str | dict
    provider: str
    model: str
    token_usage: dict
    finish_reason: str | None
    raw_response: dict | None
```

## 4.2 Execution Contracts

```python
class ExecutionContext:
    run_id: str
    workflow_id: str
    step_id: str
    actor_id: str | None
    workspace_id: str | None
    tenant_id: str | None

class StepExecutionRequest:
    workflow_definition: WorkflowDefinition
    step: WorkflowStep
    state: WorkflowState
    context_packet: ContextPacket
    tool_access: ToolAccessDecision

class StepExecutionResult:
    status: str
    output: object
    next_step: str | None
    artifacts: list[ArtifactRef]
    memory_writes: list[MemoryWriteIntent]
    events: list[RuntimeEvent]
```

## 4.3 Memory Contracts

```python
class MemoryRecord:
    record_id: str
    namespace: str
    memory_type: str
    content: str
    metadata: dict
    source_run_id: str
    source_step_id: str
    created_at: str
    expires_at: str | None

class MemoryPolicy:
    namespace: str
    allowed_types: list[str]
    max_records: int | None
    ttl_days: int | None
    consolidation_strategy: str
```

## 4.4 Context Contracts

```python
class ContextPacket:
    current_task: str
    memory_summary: str
    compacted_history: str
    recent_history: list[dict]
    retrieved_artifacts: list[ArtifactRef]
    working_notes: str
    token_budget: int | None
    context_brief: str
```

## 4.5 Tool Contracts

```python
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    input_schema: dict
    output_schema: dict | None
    capability_tags: list[str]
    sandbox_profile: str | None

class ToolExecutionRequest:
    tool_id: str
    arguments: dict
    execution_context: ExecutionContext

class ToolExecutionResult:
    status: str
    output: object
    error: str | None
    latency_ms: int | None
```

## 4.6 Governance Contracts

```python
class IdentityContext:
    principal_id: str
    tenant_id: str | None
    workspace_id: str | None
    roles: list[str]
    attributes: dict

class AuthorizationDecision:
    allowed: bool
    reason: str
    constraints: dict
```

## 4.7 Artifact Contracts

```python
class ArtifactContract:
    artifact_type: str
    format: str
    required: bool
    writer: str

class ArtifactRef:
    artifact_id: str
    artifact_type: str
    uri: str
    version: str | None
    provenance: dict
```

## 4.8 Evaluation Contracts

```python
class EvalPolicy:
    runtime_critics: list[str]
    offline_suites: list[str]
    thresholds: dict[str, float]

class EvalResult:
    evaluator_id: str
    score: float | None
    status: str
    findings: list[str]
    metadata: dict
```

---

# 5. Agent vs Workflow

This distinction should be explicit.

## Agent

An agent is a configured execution unit with:

- role/purpose
- cognitive profile
- tool bundle
- memory policy
- guardrails
- evaluation hooks

It is the “who” of a task.

## Workflow

A workflow is a runtime graph that coordinates:

- one or more agents
- tools
- memory operations
- review gates
- artifacts
- retries and branching

It is the “how” of execution.

## Rule

Agents are components inside workflows.

Short-lived agents can be instantiated per step or per subgraph.
Long-lived agents can be bound to persistent memory namespaces and durable run state.

---

# 6. Runtime Authority Model

This should govern implementation decisions.

## Execution Authority Flow

1. `Workflow Definition Layer` declares desired behavior.
2. `Governance Plane` authorizes capabilities and access.
3. `Agentic OS` executes the workflow using approved capabilities.
4. `Cognitive Layer` supplies inference and reasoning primitives.
5. `Evaluation Plane` critiques and scores runtime behavior.

This model prevents the orchestration runtime from becoming the source of truth for policy.

## Practical Example

If a workflow says:

- allowed tool: `web_search`
- memory namespace: `market_research`
- human approval required before `final_delivery`

Then:

- the workflow definition declares intent
- governance verifies tool and namespace entitlement
- OS runs the graph and pauses at the review gate
- cognitive layer executes model steps
- evaluation plane scores evidence quality

---

# 7. Workflow Definition Contracts

Workflow definitions should be declarative and contract-first.

Minimum required concepts:

- workflow metadata
- step graph
- agent assignment
- allowed tools
- memory policy reference
- context policy reference
- guardrail policy reference
- artifact outputs
- approval gates
- eval hooks

Example direction:

```yaml
name: market_research_agent

agent:
  role: research_analyst
  cognitive_profile: gpt_research_default

allowed_tools:
  - web_search
  - document_reader

memory_policy:
  namespace: market_research
  strategy: semantic_plus_episode

context_policy:
  max_memory_items: 5
  max_history_items: 4
  summarizer: compact_summary_v1

guardrails:
  - no_unverified_claims
  - cite_sources

outputs:
  - type: markdown_report
    required: true
  - type: evidence_table
    required: true

human_review:
  required: true
  before: final_delivery

evals:
  runtime:
    - factuality_score
    - source_quality_score
```

---

# 8. Incremental Migration From Current Code

The current `agentic_harness` package already contains a useful seed:

- workflow DSL parsing
- LangGraph orchestration
- filesystem memory
- context manager
- optional LLM adapter
- CLI runner

Recommended migration path:

## Phase 1: Stabilize the Current Runtime

- keep existing `agentic_harness` package operational
- rename modules toward future boundaries only when responsibilities are clear
- introduce stable core contracts first

## Phase 2: Split Into Execution Domains

First split current code into:

- `cognitive/`
- `agentic_os/orchestration/`
- `agentic_os/context/`
- `agentic_os/memory/`
- `definitions/workflows/`

Do not introduce governance/evaluation implementation before core contracts exist.

## Phase 3: Introduce Cross-Cutting Planes

Add:

- governance authorization hooks before tool/memory access
- evaluation hooks before/after important steps
- artifact contract enforcement

## Phase 4: Support Multi-Agent Workflows

Add:

- agent definitions
- agent registry
- agent instantiation policies
- subgraph-per-agent execution patterns

---

# 9. Implementation Priorities

Recommended build order:

1. `Core contracts`
2. `Workflow definitions and loaders`
3. `OS orchestration runtime`
4. `Memory manager`
5. `Context manager + compaction`
6. `Tool registry/executor`
7. `Artifact manager`
8. `Governance authorization hooks`
9. `Runtime evaluation hooks`
10. `Multi-agent coordination`

This order keeps the platform usable at every stage.

---

# 10. Non-Negotiable Design Rules

- The OS exposes services, not business behaviors.
- Governance is the source of truth for access and approval policy.
- Evaluation is a first-class platform concern, not an afterthought.
- Workflows and agents must be contract-driven.
- Artifacts are first-class outputs, not incidental files.
- Context engineering must remain inspectable.
- Memory must be namespace-aware and policy-driven.
- LLM providers must remain replaceable.

---

# 11. Immediate Next Refactor Target

The next refactor should not try to implement the full platform at once.

The best immediate target is:

- keep the current code working
- introduce `definitions/`, `agentic_os/`, and `cognitive/` top-level packages
- move existing modules into those three buckets
- keep governance and evaluation as interface-only packages first

That yields a structurally correct codebase before the heavier enterprise features are implemented.

