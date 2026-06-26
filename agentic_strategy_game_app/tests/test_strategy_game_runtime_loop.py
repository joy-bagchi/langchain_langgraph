from __future__ import annotations

from agentic_strategy_game_app.engine import StrategyGameEngine
from agentic_strategy_game_app.runtime_loop import synchronize_world_with_elapsed_time
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario
from agentic_strategy_game_app.simulation_clock import quarter_duration_seconds, simulated_date_label


def test_runtime_loop_advances_world_when_a_quarter_elapses() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    anchor = 1_000.0
    now = anchor + quarter_duration_seconds(world.current_period_label) + 1

    next_world, next_anchor, turns = synchronize_world_with_elapsed_time(
        world=world,
        clock_anchor_epoch=anchor,
        now_epoch=now,
        engine=engine,
    )

    assert next_world.current_turn == 1
    assert next_world.current_period_label == "Q2"
    assert len(turns) == 1
    assert next_anchor == anchor + quarter_duration_seconds("Q1")


def test_runtime_loop_can_advance_multiple_quarters() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    anchor = 1_000.0
    now = anchor + quarter_duration_seconds("Q1") + quarter_duration_seconds("Q2") + 10

    next_world, next_anchor, turns = synchronize_world_with_elapsed_time(
        world=world,
        clock_anchor_epoch=anchor,
        now_epoch=now,
        engine=engine,
    )

    assert next_world.current_turn == 2
    assert next_world.current_period_label == "Q3"
    assert len(turns) == 2
    assert next_anchor == anchor + quarter_duration_seconds("Q1") + quarter_duration_seconds("Q2")


def test_simulated_date_label_advances_inside_quarter() -> None:
    anchor = 500.0

    assert simulated_date_label("Q1", anchor_epoch=anchor, now_epoch=anchor) == "Jan 1"
    assert simulated_date_label("Q1", anchor_epoch=anchor, now_epoch=anchor + 15) == "Jan 4"
