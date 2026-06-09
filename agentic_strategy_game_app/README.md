# Agentic Strategy Game App

`agentic_strategy_game_app` is a separate application module built on top of
`agentic_harness`. It is the start of a turn-based strategy flight simulator
for testing business decisions under uncertainty, competition, and ecosystem
pressure.

## Current Slice

Slice 1 is implemented:

- typed simulation contracts
- scenario and world-state scaffolding
- serialization / deserialization support
- basic validation rules
- a small inspection CLI for the initial scenario
- a deterministic player-strategy interpreter for plain-English input
- a deterministic single-turn engine
- a simulation calendar in the UI that ticks through the active quarter
- autonomous quarter rollover when time elapses without player action

This slice does not yet include:

- a simulation engine
- multi-turn progression
- rule-based or LLM decision providers

## Scenario

The initial scenario is:

- `b2b_saas_ai_disruption`

It includes:

- three company actors
- enterprise customers
- capital markets
- a regulator
- market forces aligned with AI-driven disruption in B2B SaaS

## Run The CLI

From the repo root:

```bash
python -m agentic_strategy_game_app.cli list-scenarios
```

```bash
python -m agentic_strategy_game_app.cli describe-scenario --name b2b_saas_ai_disruption
```

## Streamlit Frontend

You can inspect the scenario through a Streamlit dashboard. Install the UI
dependency first:

```bash
pip install streamlit
```

Then launch it from the repo root:

```bash
streamlit run agentic_strategy_game_app/streamlit_app.py
```

The current UI supports:

- scenario selection
- market-force perturbation controls
- market pressure summary cards
- company comparison tables
- ecosystem actor inspection
- a correlation and diagnostics view
- a player strategy panel that converts plain-English intent into structured actions
- a deterministic quarter-advance loop for player actions
- a top-right simulation calendar with in-quarter date progression
- autonomous world progression when the quarter expires

The UI now supports a deterministic one-turn simulation loop. Multi-turn
ecosystem reactions and non-player agents still come next.

## Files

- `contracts.py` defines the typed simulation kernel
- `scenarios.py` defines the initial scenario scaffold
- `dashboard.py` defines pure dashboard helpers
- `player_strategy.py` defines the deterministic player strategy interpreter
- `engine.py` defines the deterministic single-turn simulation engine
- `simulation_clock.py` defines the simulation calendar helpers
- `runtime_loop.py` defines elapsed-time synchronization for the simulation
- `cli.py` exposes a small inspection CLI
- `streamlit_app.py` exposes the first operator-facing UI
- `tests/` contains contract and scenario tests
