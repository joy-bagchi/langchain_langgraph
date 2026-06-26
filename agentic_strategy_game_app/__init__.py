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
    VCPitchAgentResponse,
    VCPitchDecision,
    VCPitchSessionState,
    VCPitchTurn,
    WorldState,
)
from agentic_strategy_game_app.app_runtime import default_vc_agent_path, run_vc_investor_agent
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
from agentic_strategy_game_app.vc_pitch import (
    append_player_pitch_message,
    apply_vc_agent_response,
    build_vc_agent_input_payload,
    create_vc_pitch_session,
    extract_vc_agent_output_payload,
    parse_vc_agent_response,
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
    "VCPitchAgentResponse",
    "VCPitchDecision",
    "VCPitchSessionState",
    "VCPitchTurn",
    "WorldState",
    "default_vc_agent_path",
    "run_vc_investor_agent",
    "StrategyGameEngine",
    "interpret_player_strategy",
    "synchronize_world_with_elapsed_time",
    "SimulationCalendarPayload",
    "build_simulation_calendar_payload",
    "simulated_date_label",
    "append_player_pitch_message",
    "apply_vc_agent_response",
    "build_vc_agent_input_payload",
    "create_vc_pitch_session",
    "extract_vc_agent_output_payload",
    "parse_vc_agent_response",
    "build_b2b_saas_ai_disruption_scenario",
    "list_scenarios",
]
