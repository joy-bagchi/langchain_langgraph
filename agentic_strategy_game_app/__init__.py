"""Agentic business strategy game app built on top of agentic_harness."""

from agentic_strategy_game_app.contracts import (
    AgentObservation,
    AgentProfile,
    CompanyState,
    GameEvent,
    MarketForces,
    PendingEffect,
    SimulationConfig,
    StrategyInterpretation,
    StrategicAction,
    TurnResult,
    UserStrategyIntent,
    WorldState,
)
from agentic_strategy_game_app.engine import StrategyGameEngine
from agentic_strategy_game_app.player_strategy import interpret_player_strategy
from agentic_strategy_game_app.runtime_loop import synchronize_world_with_elapsed_time
from agentic_strategy_game_app.scenarios import (
    build_b2b_saas_ai_disruption_scenario,
    list_scenarios,
)
from agentic_strategy_game_app.simulation_clock import (
    SimulationCalendarPayload,
    build_simulation_calendar_payload,
    simulated_date_label,
)

__all__ = [
    "AgentObservation",
    "AgentProfile",
    "CompanyState",
    "GameEvent",
    "MarketForces",
    "PendingEffect",
    "SimulationConfig",
    "StrategyInterpretation",
    "StrategicAction",
    "TurnResult",
    "UserStrategyIntent",
    "WorldState",
    "StrategyGameEngine",
    "interpret_player_strategy",
    "synchronize_world_with_elapsed_time",
    "SimulationCalendarPayload",
    "build_simulation_calendar_payload",
    "simulated_date_label",
    "build_b2b_saas_ai_disruption_scenario",
    "list_scenarios",
]
