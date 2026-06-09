from __future__ import annotations

from agentic_strategy_game_app.engine import StrategyGameEngine
from agentic_strategy_game_app.player_strategy import interpret_player_strategy
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario


def test_single_turn_engine_applies_cut_price() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    actions = interpret_player_strategy(
        world=world,
        actor_id="incumbent_platform",
        raw_strategy="Cut price aggressively this quarter.",
        max_actions=2,
    ).actions

    next_world, turn_result = engine.run_turn(world, actions)

    assert turn_result.turn_number == 1
    assert turn_result.period_label == "Q1"
    assert next_world.current_period_label == "Q2"
    assert next_world.companies["incumbent_platform"].market_share > world.companies["incumbent_platform"].market_share
    assert next_world.companies["incumbent_platform"].margin < world.companies["incumbent_platform"].margin
    assert turn_result.actions_rejected == []


def test_single_turn_engine_rejects_action_when_cash_is_insufficient() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    actions = interpret_player_strategy(
        world=world,
        actor_id="ai_native_startup",
        raw_strategy="Invest heavily in AI and expand sales aggressively.",
        max_actions=2,
    ).actions
    actions[0].resource_cost = world.companies["ai_native_startup"].cash + 1.0

    next_world, turn_result = engine.run_turn(world, actions)

    assert turn_result.actions_rejected
    assert "Insufficient cash" in turn_result.actions_rejected[0]["reason"]
    assert next_world.companies["ai_native_startup"].ai_capability >= world.companies["ai_native_startup"].ai_capability


def test_single_turn_engine_updates_rnd_and_operations() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    actions = interpret_player_strategy(
        world=world,
        actor_id="enterprise_suite_competitor",
        raw_strategy="Invest in AI and improve operations carefully.",
        max_actions=2,
    ).actions

    next_world, turn_result = engine.run_turn(world, actions)

    company = next_world.companies["enterprise_suite_competitor"]
    baseline = world.companies["enterprise_suite_competitor"]

    assert company.ai_capability > baseline.ai_capability
    assert company.operational_efficiency > baseline.operational_efficiency
    assert company.technical_debt < baseline.technical_debt
    assert len(turn_result.actions_accepted) == 2


def test_single_turn_engine_enforces_max_actions_per_turn() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=1)
    interpretation = interpret_player_strategy(
        world=world,
        actor_id="incumbent_platform",
        raw_strategy="Cut price, invest in AI, and increase marketing.",
        max_actions=3,
    )

    _, turn_result = engine.run_turn(world, interpretation.actions)

    assert len(turn_result.actions_accepted) == 1
    assert len(turn_result.actions_rejected) == 2


def test_single_turn_engine_applies_passive_drift_without_player_actions() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)

    next_world, turn_result = engine.run_turn(world, [])

    assert next_world.current_turn == 1
    assert next_world.current_period_label == "Q2"
    assert turn_result.actions_accepted == []
    assert turn_result.actions_rejected == []
    assert "passive_dynamics" in turn_result.market_results
    assert turn_result.narrative_summary.startswith("No player actions were proposed")


def test_player_action_does_not_advance_quarter_when_time_is_not_advanced() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    engine = StrategyGameEngine(max_actions_per_turn=2)
    actions = interpret_player_strategy(
        world=world,
        actor_id="incumbent_platform",
        raw_strategy="Invest in AI and increase marketing.",
        max_actions=2,
    ).actions

    next_world, turn_result = engine.run_turn(world, actions, advance_time=False)

    assert next_world.current_turn == world.current_turn
    assert next_world.current_period_label == world.current_period_label
    assert turn_result.turn_number == world.current_turn
    assert "Accepted in-quarter actions" in turn_result.narrative_summary
