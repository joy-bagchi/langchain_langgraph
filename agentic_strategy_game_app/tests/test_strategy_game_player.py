from __future__ import annotations

from agentic_strategy_game_app.player_strategy import interpret_player_strategy
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario


def test_player_strategy_interpreter_extracts_actions_and_systems() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()

    interpretation = interpret_player_strategy(
        world=world,
        actor_id="incumbent_platform",
        raw_strategy="Cut price moderately, invest heavily in AI, and push marketing harder this quarter.",
        max_actions=3,
    )

    action_types = [action.action_type for action in interpretation.actions]

    assert action_types == ["cut_price", "increase_rnd", "increase_marketing"]
    assert "pricing" in interpretation.internal_systems
    assert "product" in interpretation.internal_systems
    assert "customers" in interpretation.external_systems
    assert not interpretation.warnings


def test_player_strategy_interpreter_warns_when_too_many_actions_are_requested() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()

    interpretation = interpret_player_strategy(
        world=world,
        actor_id="ai_native_startup",
        raw_strategy="Invest in AI, expand sales, increase marketing, and form a partnership with a hyperscaler.",
        max_actions=2,
    )

    assert len(interpretation.actions) == 2
    assert interpretation.warnings
    assert "first 2 are kept" in interpretation.warnings[0]


def test_player_strategy_interpreter_handles_unrecognized_input() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()

    interpretation = interpret_player_strategy(
        world=world,
        actor_id="enterprise_suite_competitor",
        raw_strategy="Stay sharp and be thoughtful.",
        max_actions=2,
    )

    assert interpretation.actions == []
    assert interpretation.warnings
