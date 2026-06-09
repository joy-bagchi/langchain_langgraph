# Agentic Business Strategy Game — Codex Implementation Plan

## 1. Purpose

Build an **agentic business strategy game** on top of the existing `agentic_harness` platform.

The game should function as a **Strategy Flight Simulator**: a turn-based simulation where AI agents represent firms, competitors, customers, regulators, capital markets, suppliers, labor markets, and other ecosystem actors. Each agent observes the world, chooses actions based on incentives and beliefs, and causes the strategy landscape to evolve over time.

The goal is not to build a static business framework visualizer. The goal is to create a dynamic strategy laboratory where users can test strategic decisions under uncertainty, competitive reaction, internal constraints, and time-delayed consequences.

---

## 2. Core Design Philosophy

### 2.1 Strategy Is Dynamic, Not Static

Business strategy is not a one-time decision. It is a repeated loop:

```text
Observe environment
→ Interpret forces
→ Choose actions
→ Competitors and ecosystem react
→ State changes
→ New constraints emerge
→ Choose again
```

The game should model strategy as repeated adaptation over time.

### 2.2 Time Is the Unit of Progress

The primary unit of progress is a **turn**.

Recommended default:

```text
1 turn = 1 business quarter
```

Every turn should produce:

1. Updated external environment
2. Agent observations
3. Agent actions
4. Validated strategic moves
5. Payoff calculations
6. State transitions
7. Narrative explanation
8. Emerging risks and opportunities

Time matters because strategic actions often have delayed effects. For example:

- R&D investment may not improve product quality until later turns.
- Brand damage may persist for multiple turns.
- Regulatory scrutiny may increase slowly before triggering enforcement.
- Capital availability may tighten during macroeconomic shocks.
- Pricing actions may produce short-term growth but long-term margin compression.

### 2.3 Agents Propose; Engine Disposes

LLM or AI agents should **not directly mutate the world state**.

Instead:

```text
Agent observes state
→ Agent proposes action
→ Rules engine validates action
→ Simulation engine applies action
→ Payoff model computes results
→ World state is updated
```

This keeps the simulation reproducible, testable, and controllable.

### 2.4 Partial Information Is Essential

Agents should not necessarily see the full world state. Real firms operate under incomplete information.

Each agent should receive an `AgentObservation`, not the full `WorldState`.

Example:

```text
True competitor cash: $500M
Observed by player: "Competitor appears well-capitalized; confidence 70%."
```

This enables strategic ambiguity, bluffing, misinterpretation, signaling, and surprise.

### 2.5 Strategy Should Be Explainable

Every turn should produce an executive-style narrative explaining:

- What happened
- Why it happened
- Which forces changed
- Which agent actions mattered
- What second-order effects are emerging
- What strategic decisions are now available

The game should train strategic judgment, not merely produce numerical outputs.

---

## 3. High-Level Architecture

Add the strategy game as a new subsystem within `agentic_harness`.

Suggested architecture:

```text
agentic_harness/
  strategy_game/
    __init__.py
    models.py
    engine.py
    actions.py
    agents.py
    observations.py
    payoffs.py
    events.py
    scenarios.py
    reporter.py
    cli.py
    tests/
```

Adapt names and structure to match the existing repo conventions.

### 3.1 Main Components

```text
StrategyGameEngine
  ├── ScenarioLoader
  ├── WorldState
  ├── MarketForces
  ├── CompanyState
  ├── AgentProfile
  ├── ObservationBuilder
  ├── AgentDecisionProvider
  ├── ActionValidator
  ├── ActionResolver
  ├── PayoffModel
  ├── EventSystem
  ├── TurnRunner
  └── NarrativeReporter
```

---

## 4. Core Domain Concepts

### 4.1 WorldState

Represents the full simulation state.

Should include:

```text
current_turn
current_period_label
scenario_name
market_forces
companies
ecosystem_agents
active_events
historical_results
pending_effects
```

### 4.2 MarketForces

External environment variables.

Examples:

```text
market_size
growth_rate
customer_budget_index
rivalry_intensity
buyer_power
supplier_power
threat_of_new_entry
threat_of_substitution
regulatory_pressure
capital_availability
talent_availability
technology_shift_intensity
macroeconomic_pressure
```

These can map loosely to strategic frameworks such as Porter’s Five Forces, ecosystem strategy, platform economics, and macro environment analysis.

### 4.3 CompanyState

Internal state of each company.

Examples:

```text
cash
revenue
margin
profit
market_share
customer_count
product_quality
ai_capability
brand_trust
sales_capacity
operational_efficiency
talent_density
technical_debt
regulatory_risk
strategic_momentum
```

### 4.4 AgentProfile

Defines how an agent behaves.

Fields may include:

```text
agent_id
name
agent_type
objective_function
risk_tolerance
cooperation_tendency
aggression_level
price_sensitivity
innovation_bias
capital_discipline
regulatory_sensitivity
time_horizon
memory_depth
```

Agent types:

```text
company
competitor
customer_segment
regulator
capital_market
supplier
labor_market
media
```

### 4.5 StrategicAction

An action proposed by an agent.

Examples:

```text
cut_price
raise_price
increase_rnd
launch_ai_feature
expand_sales_capacity
reduce_headcount
acquire_company
form_partnership
enter_market
exit_market
increase_marketing
improve_operations
lobby_regulator
bundle_products
open_platform
close_platform
```

Each action should have:

```text
action_id
actor_id
action_type
intensity
resource_cost
expected_effects
constraints
rationale
```

### 4.6 TurnResult

Output from one turn.

Should include:

```text
turn_number
period_label
actions_submitted
actions_accepted
actions_rejected
state_before
state_after
financial_results
market_results
agent_payoffs
triggered_events
pending_effects_created
narrative_summary
```

---

## 5. Agentic Subsystems

### 5.1 Observation Subsystem

Responsible for converting full world state into agent-specific observations.

```text
WorldState → ObservationBuilder → AgentObservation
```

Observation should include:

```text
visible_market_forces
visible_competitor_signals
own_company_state
known_events
uncertain_beliefs
memory_summary
available_actions
```

Different agents may see different things.

Example:

- Capital market agent sees financial discipline and growth trajectory.
- Regulator agent sees market concentration and consumer harm.
- Customer agent sees price, quality, trust, and switching cost.
- Competitor sees market share signals but not exact internal cost structure.

### 5.2 Decision Subsystem

Responsible for producing proposed actions.

Define an interface:

```text
AgentDecisionProvider
  decide(observation, agent_profile, game_rules) -> list[StrategicAction]
```

Implement in phases:

1. `RuleBasedAgentDecisionProvider`
2. `ScriptedAgentDecisionProvider`
3. `LlmAgentDecisionProvider`

The LLM provider should provide rationale and proposed actions, but should not apply effects.

### 5.3 Validation Subsystem

Responsible for determining whether actions are legal and feasible.

Examples:

- Cannot acquire a company without enough cash or financing.
- Cannot cut price below zero.
- Cannot launch AI feature without minimum AI capability.
- Cannot form cartel without triggering legal/regulatory risk.
- Cannot expand sales faster than talent availability allows.

### 5.4 Resolution Subsystem

Responsible for applying valid actions to the world.

The same action may have different effects depending on context.

Example:

```text
cut_price
```

Effects may depend on:

- buyer power
- customer price sensitivity
- competitor response
- brand trust
- margin structure
- market maturity

### 5.5 Payoff Subsystem

Responsible for computing outcomes.

Payoffs should include both financial and strategic consequences.

Examples:

```text
revenue
profit
cash
market_share
customer_growth
brand_trust
regulatory_risk
strategic_position
investor_confidence
```

### 5.6 Memory Subsystem

Agents should remember prior interactions.

Examples:

- Competitor remembers repeated price aggression.
- Regulator remembers repeated anti-competitive behavior.
- Customers remember broken promises or quality failures.
- Capital markets remember missed growth expectations.

Memory should influence future decisions.

### 5.7 Event Subsystem

Events introduce external shocks or endogenous consequences.

Event types:

```text
macro_shock
technology_breakthrough
regulatory_investigation
new_entrant
supply_constraint
labor_shortage
customer_budget_cut
capital_market_tightening
reputation_crisis
```

Events may be:

```text
random
scripted
triggered by state thresholds
triggered by agent behavior
```

### 5.8 Narrative Reporter

Produces executive-facing explanation after each turn.

Required format:

```text
Turn Summary
- What happened
- Why it happened
- Key metric changes
- Strategic implications
- Emerging risks
- Recommended next decision
```

---

## 6. Build Slices for Codex

Build in thin vertical slices. Do not attempt to build the full simulation at once.

---

## Slice 0 — Repository Inspection

### Goal

Understand `agentic_harness` before making changes.

### Codex Task

```text
Inspect the repository and summarize the existing architecture.

Identify:
1. Existing agent abstractions
2. Existing orchestration or workflow loops
3. Existing state/memory patterns
4. Existing tool or plugin interfaces
5. Existing CLI or entry points
6. Existing testing conventions
7. Best location to add a strategy game subsystem

Do not write code yet. Produce an implementation plan that fits the current repo style.
```

### Acceptance Criteria

- No code changes.
- Clear summary of existing architecture.
- Recommendation for where the new subsystem should live.
- Identification of reusable abstractions from `agentic_harness`.

---

## Slice 1 — Domain Models

### Goal

Create the typed data model for the game.

### Codex Task

```text
Add a new strategy game module using the repo's conventions.

Create typed domain models for:
- WorldState
- MarketForces
- CompanyState
- AgentProfile
- AgentObservation
- StrategicAction
- TurnResult
- SimulationConfig
- PendingEffect
- GameEvent

Do not integrate LLM calls yet.
Add unit tests for model creation, serialization, and basic validation.
```

### Acceptance Criteria

- Models are typed.
- Models can be serialized/deserialized if the repo supports that pattern.
- Tests pass.
- No LLM calls.
- No complex game logic yet.

---

## Slice 2 — Deterministic Single-Turn Engine

### Goal

Create a basic engine that can process one turn deterministically.

### Codex Task

```text
Implement a deterministic StrategyGameEngine that can run one turn.

The engine should:
1. Accept a WorldState and a list of StrategicAction objects
2. Validate actions
3. Apply valid actions
4. Reject invalid actions with reasons
5. Compute basic financial and market outcomes
6. Return a TurnResult

Implement only a small initial action set:
- cut_price
- raise_price
- increase_rnd
- increase_marketing
- improve_operations

Add tests for each action.
```

### Acceptance Criteria

- One turn can run without LLMs.
- Invalid actions are rejected safely.
- State before and after are captured.
- Tests cover each action.

---

## Slice 3 — Multi-Turn Simulation and Time

### Goal

Make time explicit.

### Codex Task

```text
Extend StrategyGameEngine to run multiple turns.

Add:
- current_turn tracking
- period labels such as Q1, Q2, Q3, Q4
- historical results
- pending effects that mature after N turns

Examples:
- R&D investment improves product quality after 2 turns
- Marketing increases customer acquisition during the next turn
- Operations improvement increases margin after 1 turn

Add tests for delayed effects and multi-turn progression.
```

### Acceptance Criteria

- Simulation can run for N turns.
- Turn count increments correctly.
- Pending effects apply at the correct future turn.
- History is preserved.

---

## Slice 4 — Scenario Loader

### Goal

Create reusable game scenarios.

### Codex Task

```text
Add a ScenarioLoader for predefined simulation scenarios.

Create one initial scenario:
- b2b_saas_ai_disruption

The scenario should include:
- market forces
- 3 company agents
- initial company states
- available actions
- default game rules

Add tests that the scenario loads and can run for at least 4 turns using scripted actions.
```

### Acceptance Criteria

- Scenario can be loaded by name.
- Scenario initializes a valid WorldState.
- Scenario can run through multiple turns.

---

## Slice 5 — Rule-Based Agents

### Goal

Add agents that can choose actions without LLMs.

### Codex Task

```text
Implement AgentDecisionProvider interface.

Create RuleBasedAgentDecisionProvider with several strategic personalities:
- AggressiveGrowthAgent
- MarginDefenderAgent
- InnovationLeaderAgent
- CapitalDisciplineAgent
- PriceWarAgent

Each agent should inspect its AgentObservation and choose actions.

Integrate agents into the multi-turn engine so the simulation can run without manually supplied actions.
```

### Acceptance Criteria

- Agents can choose actions from observations.
- Different personalities produce meaningfully different behavior.
- Simulation can run 8+ turns using only rule-based agents.
- Tests verify at least two agents choose different actions in the same situation.

---

## Slice 6 — Observation Builder and Partial Information

### Goal

Prevent agents from seeing the full world state.

### Codex Task

```text
Implement ObservationBuilder.

It should convert WorldState into AgentObservation based on:
- agent type
- visibility rules
- uncertainty settings
- known history

Agents should receive observations rather than direct access to WorldState.

Add tests verifying that:
- a company sees its own exact state
- a company sees only estimates of competitor state
- regulator sees regulatory-risk indicators
- customer segment sees price, quality, and trust indicators
```

### Acceptance Criteria

- Agents no longer require full WorldState.
- Observations differ by agent type.
- Tests validate hidden information behavior.

---

## Slice 7 — Event System

### Goal

Add external shocks and triggered events.

### Codex Task

```text
Implement GameEvent and EventSystem.

Support events that are:
- scripted by turn number
- random using a seeded RNG
- triggered by thresholds

Initial events:
- recession_shock
- ai_breakthrough
- regulatory_investigation
- capital_market_tightening
- new_entrant

Add tests using seeded randomness for deterministic behavior.
```

### Acceptance Criteria

- Events can be scripted.
- Events can be seeded for deterministic tests.
- Events can modify market forces or company states.
- Triggered events occur when thresholds are crossed.

---

## Slice 8 — Strategic Payoff Model

### Goal

Make payoffs more strategy-relevant.

### Codex Task

```text
Enhance PayoffModel.

Compute:
- revenue
- profit
- cash
- market_share
- customer_growth
- brand_trust
- investor_confidence
- regulatory_risk
- strategic_momentum

The model should reflect interaction effects.

Examples:
- Price cuts increase share more when buyer power is high.
- R&D matters more when technology_shift_intensity is high.
- Aggressive growth consumes cash but may improve market share.
- Regulatory risk rises when market concentration and anti-competitive behavior increase.
```

### Acceptance Criteria

- Outcomes are sensitive to market forces.
- Interaction effects are tested.
- Payoff calculations remain deterministic for tests.

---

## Slice 9 — LLM Agent Decision Provider

### Goal

Allow LLM agents to propose strategic actions.

### Codex Task

```text
Implement LlmAgentDecisionProvider.

The provider should:
1. Receive AgentObservation, AgentProfile, and available actions
2. Ask the LLM to choose actions and explain rationale
3. Parse the response into StrategicAction proposals
4. Return proposed actions to the engine

Important:
- The LLM must not directly mutate WorldState.
- The engine must validate all actions.
- The provider should support mock mode for tests.
- Add robust parsing and fallback behavior.
```

### Acceptance Criteria

- LLM provider conforms to AgentDecisionProvider.
- Mock provider tests pass without external API calls.
- Invalid LLM actions are rejected by validator.
- Rationale is preserved for reporting.

---

## Slice 10 — Cooperation, Collusion, and Regulatory Risk

### Goal

Model cooperation and cartel-like behavior as risky strategic dynamics.

### Codex Task

```text
Add support for cooperative and anti-competitive actions.

Actions may include:
- form_partnership
- signal_pricing_discipline
- tacit_coordination
- exclusive_supplier_agreement

Important:
- Explicitly label cartel/collusive behavior as illegal or anti-competitive in the simulation.
- Model regulatory detection and enforcement risk.
- Do not encourage real-world illegal conduct.
- Treat this as simulation behavior with consequences.
```

### Acceptance Criteria

- Cooperative behavior can create short-term payoff advantages.
- Anti-competitive behavior increases regulatory risk.
- Regulatory investigation can be triggered.
- Narrative reporter flags illegal/anti-competitive behavior clearly.

---

## Slice 11 — Narrative Reporter

### Goal

Make the simulation useful for executive strategy training.

### Codex Task

```text
Implement NarrativeReporter.

For each turn, generate:
- Turn Summary
- Key Actions
- Market Response
- Financial Results
- Strategic Implications
- Emerging Risks
- Recommended Next Decision

For full simulation, generate:
- Strategy arc
- Winning and losing moves
- Major inflection points
- Lessons learned
```

### Acceptance Criteria

- Reporter can generate a summary from TurnResult.
- Reporter does not require LLM for basic output.
- Optional LLM enhancement can be added later.
- Output is concise and executive-readable.

---

## Slice 12 — CLI Runner

### Goal

Allow the game to run from the command line.

### Codex Task

```text
Add a CLI entry point for the strategy game.

Example usage:

strategy-game run --scenario b2b_saas_ai_disruption --turns 12 --agents rule_based

Output per turn:
- actions
- key metric changes
- events
- narrative summary

Also support JSON output:

strategy-game run --scenario b2b_saas_ai_disruption --turns 12 --output json
```

### Acceptance Criteria

- CLI can run a scenario.
- CLI can run multiple turns.
- CLI prints readable output.
- CLI supports machine-readable output.

---

## Slice 13 — Persistence and Replay

### Goal

Allow simulations to be saved, inspected, and replayed.

### Codex Task

```text
Add persistence support for simulation runs.

The system should save:
- initial config
- random seed
- all observations
- all proposed actions
- accepted/rejected actions
- turn results
- final state

Add replay support so a prior run can be reproduced.
```

### Acceptance Criteria

- Simulation run can be saved.
- Simulation run can be replayed deterministically.
- Replay produces same results when using same seed and same actions.

---

## Slice 14 — Evaluation Harness

### Goal

Compare strategies across repeated runs.

### Codex Task

```text
Add an evaluation harness that can run many simulations and compare outcomes.

Support:
- repeated runs with different seeds
- strategy A vs strategy B comparisons
- summary metrics
- win/loss analysis
- sensitivity to market conditions
```

### Acceptance Criteria

- Can run multiple simulations in batch.
- Can compare two agent strategies.
- Produces aggregate summary metrics.

---

## 7. Initial MVP Definition

The MVP is complete when the system can run:

```text
strategy-game run --scenario b2b_saas_ai_disruption --turns 12 --agents rule_based
```

And produce:

1. A multi-turn simulation
2. Multiple agents choosing actions
3. Market and company state changes
4. Delayed effects over time
5. At least one external event
6. A narrative summary per turn
7. Final simulation summary

No UI is required for MVP.

---

## 8. Recommended First Scenario: B2B SaaS AI Disruption

### Market Context

An established B2B SaaS market is being disrupted by AI-native entrants.

### Agents

```text
Incumbent Platform
AI-Native Startup
Enterprise Suite Competitor
Customer Segment
Capital Market
Regulator
```

### Initial Conditions

```text
market_size: high
growth_rate: moderate
buyer_power: medium
supplier_power: low
rivalry_intensity: medium
technology_shift_intensity: high
capital_availability: medium
regulatory_pressure: low
```

### Strategic Tension

The incumbent has scale, customers, brand, and cash, but also technical debt. The AI-native startup has superior AI capability and speed, but limited cash and distribution. The enterprise suite competitor has bundling power and customer relationships.

### Strategic Questions the Game Should Surface

- Should the incumbent defend margin or accelerate AI investment?
- Can the AI-native startup grow before capital runs out?
- Does bundling by the enterprise suite competitor reduce market openness?
- Do price cuts expand adoption or destroy category profitability?
- When does regulator attention become material?
- When does technical debt become a strategic liability?

---

## 9. Non-Goals for Initial Build

Do not build these initially:

```text
Full graphical UI
Complex economic model
Real company data ingestion
Multiplayer networking
Unbounded free-form agent actions
Perfect realism
```

Prioritize:

```text
clear simulation kernel
typed models
testable rules
deterministic replay
explainable outcomes
```

---

## 10. Engineering Guardrails

1. Keep simulation deterministic where possible.
2. Use seeded randomness for tests.
3. Keep LLM calls behind interfaces.
4. Do not let LLMs mutate state directly.
5. Store state transitions explicitly.
6. Preserve action rationale for explainability.
7. Make all major calculations testable.
8. Prefer simple models first; add complexity only after the loop works.
9. Build thin vertical slices.
10. Keep CLI usable before building UI.

---

## 11. Suggested Codex Operating Mode

For each slice, ask Codex to:

1. Inspect relevant files first.
2. Propose implementation plan.
3. Make the smallest coherent change.
4. Add or update tests.
5. Run tests.
6. Summarize changed files.
7. Explain how to run the new capability.

Use this prompt pattern:

```text
You are working in the agentic_harness repo.

Implement Slice X from agentic_strategy_game_codex_plan.md.

Before coding:
- Inspect the repo structure.
- Identify relevant existing abstractions.
- Explain your implementation plan.

Then:
- Implement the smallest useful version.
- Add tests.
- Run tests.
- Summarize changed files and how to run the feature.

Do not introduce unnecessary dependencies.
Do not build future slices early unless required for this slice.
```

---

## 12. Final Product Vision

The end-state product is an agentic strategy simulator where users can test strategic moves in a controlled, dynamic, multi-agent business environment.

It should help users practice:

```text
- reading market topology
- identifying strategic levers
- anticipating competitor reaction
- understanding second-order effects
- managing internal constraints
- making decisions under uncertainty
- explaining strategy in executive language
```

This is not merely a game. It is a training environment for strategic judgment.
