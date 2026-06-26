"""Simulation-time synchronization helpers for autonomous world progression."""

from __future__ import annotations

from agentic_strategy_game_app.contracts import TurnResult, WorldState
from agentic_strategy_game_app.engine import StrategyGameEngine
from agentic_strategy_game_app.simulation_clock import quarter_duration_seconds


def synchronize_world_with_elapsed_time(
    *,
    world: WorldState,
    clock_anchor_epoch: float,
    now_epoch: float,
    engine: StrategyGameEngine,
) -> tuple[WorldState, float, list[TurnResult]]:
    """Advance the world for every fully elapsed quarter since the anchor."""
    current_world = WorldState.from_dict(world.to_dict())
    current_anchor = float(clock_anchor_epoch)
    generated_turns: list[TurnResult] = []

    while now_epoch - current_anchor >= quarter_duration_seconds(current_world.current_period_label):
        elapsed_quarter_seconds = quarter_duration_seconds(current_world.current_period_label)
        current_world, turn_result = engine.run_turn(current_world, [])
        generated_turns.append(turn_result)
        current_anchor += elapsed_quarter_seconds

    return current_world, current_anchor, generated_turns
