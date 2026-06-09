from __future__ import annotations

import json

import pytest

from agentic_strategy_game_app.cli import _render_scenario
from agentic_strategy_game_app.contracts import (
    AgentObservation,
    GameEvent,
    MarketForces,
    PendingEffect,
    StrategicAction,
    TurnResult,
    WorldState,
)
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario, list_scenarios


def test_b2b_saas_ai_disruption_scenario_builds_valid_contracts() -> None:
    config, world = build_b2b_saas_ai_disruption_scenario()

    assert config.scenario_name == "b2b_saas_ai_disruption"
    assert world.scenario_name == "b2b_saas_ai_disruption"
    assert set(world.companies) == {
        "incumbent_platform",
        "ai_native_startup",
        "enterprise_suite_competitor",
    }
    assert "capital_market" in world.ecosystem_agents
    assert world.market_forces.technology_shift_intensity > 0.8


def test_world_state_round_trips_through_dict_serialization() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()

    serialized = world.to_dict()
    restored = WorldState.from_dict(serialized)

    assert restored.to_dict() == serialized
    assert restored.companies["incumbent_platform"].name == "Incumbent Platform"


def test_market_forces_validation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        MarketForces(
            market_size=1_000_000.0,
            growth_rate=0.1,
            customer_budget_index=1.2,
            rivalry_intensity=0.5,
            buyer_power=0.4,
            supplier_power=0.4,
            threat_of_new_entry=0.3,
            threat_of_substitution=0.3,
            regulatory_pressure=0.2,
            capital_availability=0.5,
            talent_availability=0.5,
            technology_shift_intensity=0.6,
            macroeconomic_pressure=0.2,
        )


def test_observation_and_action_validation_round_trip() -> None:
    observation = AgentObservation(
        agent_id="incumbent_platform",
        turn_number=1,
        period_label="Q2",
        visible_market_forces={"rivalry_intensity": 0.6},
        visible_company_state={"cash": 1_000_000.0},
        confidence_by_signal={"competitor_capitalization": 0.7},
        available_actions=["increase_rnd", "cut_price"],
    )
    action = StrategicAction(
        action_id="action-1",
        actor_id="incumbent_platform",
        action_type="increase_rnd",
        intensity=0.65,
        resource_cost=25_000_000.0,
        rationale="Protect the core franchise against AI-native disruption.",
    )

    assert AgentObservation.from_dict(observation.to_dict()).to_dict() == observation.to_dict()
    assert StrategicAction.from_dict(action.to_dict()).to_dict() == action.to_dict()


def test_turn_result_serializes_nested_events_and_effects() -> None:
    event = GameEvent(
        event_id="event-1",
        event_type="ai_breakthrough",
        title="AI breakthrough",
        description="A foundation-model vendor cuts inference costs sharply.",
        severity=0.7,
        turn_triggered=1,
        source="scripted",
    )
    effect = PendingEffect(
        effect_id="effect-1",
        actor_id="ai_native_startup",
        effect_type="delayed_product_quality_gain",
        applies_on_turn=3,
        payload={"product_quality_delta": 0.08},
        created_turn=1,
        description="R&D investment matures in two turns.",
    )
    result = TurnResult(
        turn_number=1,
        period_label="Q1",
        actions_submitted=[{"action_id": "a-1"}],
        actions_accepted=[{"action_id": "a-1"}],
        actions_rejected=[],
        state_before={"market_share": 0.06},
        state_after={"market_share": 0.07},
        financial_results={"revenue_delta": 2_000_000.0},
        market_results={"share_shift": 0.01},
        agent_payoffs={"ai_native_startup": 0.14},
        triggered_events=[event.to_dict()],
        pending_effects_created=[effect.to_dict()],
        narrative_summary="The startup pushed R&D despite cash constraints.",
    )

    restored = TurnResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()


def test_cli_renders_json_scenario_payload() -> None:
    rendered = _render_scenario("b2b_saas_ai_disruption")

    assert rendered["simulation_config"]["scenario_name"] == "b2b_saas_ai_disruption"
    assert rendered["initial_world_state"]["companies"]["ai_native_startup"]["ai_capability"] > 0.9
    json.dumps(rendered)


def test_list_scenarios_contains_initial_scenario() -> None:
    assert "b2b_saas_ai_disruption" in list_scenarios()
