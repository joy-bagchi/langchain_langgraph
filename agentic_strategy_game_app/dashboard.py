"""Pure dashboard helpers for the strategy game app."""

from __future__ import annotations

import math

from agentic_strategy_game_app.contracts import MarketForces, WorldState
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario


def load_scenario_world(name: str = "b2b_saas_ai_disruption") -> tuple[dict, WorldState]:
    """Return the current scenario config and initial world state."""
    if name != "b2b_saas_ai_disruption":
        raise ValueError(f"Unknown scenario '{name}'.")
    config, world = build_b2b_saas_ai_disruption_scenario()
    return config.to_dict(), world


def apply_market_force_overrides(world: WorldState, overrides: dict[str, float] | None = None) -> WorldState:
    """Return a cloned world state with market-force overrides applied."""
    if not overrides:
        return WorldState.from_dict(world.to_dict())
    payload = world.to_dict()
    market_forces = dict(payload["market_forces"])
    market_forces.update({key: float(value) for key, value in overrides.items()})
    payload["market_forces"] = market_forces
    return WorldState.from_dict(payload)


def market_force_rows(world: WorldState) -> list[dict[str, float | str]]:
    forces = world.market_forces.to_dict()
    return [
        {"force": key.replace("_", " ").title(), "value": float(value)}
        for key, value in forces.items()
    ]


def company_rows(world: WorldState) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for company in world.companies.values():
        rows.append(
            {
                "company": company.name,
                "cash_m": round(company.cash / 1_000_000, 1),
                "revenue_m": round(company.revenue / 1_000_000, 1),
                "margin_pct": round(company.margin * 100, 1),
                "market_share_pct": round(company.market_share * 100, 1),
                "customers": company.customer_count,
                "product_quality": round(company.product_quality, 2),
                "ai_capability": round(company.ai_capability, 2),
                "brand_trust": round(company.brand_trust, 2),
                "technical_debt": round(company.technical_debt, 2),
                "strategic_momentum": round(company.strategic_momentum, 2),
            }
        )
    return rows


def ecosystem_rows(world: WorldState) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for agent in world.ecosystem_agents.values():
        rows.append(
            {
                "name": agent.name,
                "type": agent.agent_type,
                "objective": agent.objective_function,
                "risk_tolerance": round(agent.risk_tolerance, 2),
                "aggression": round(agent.aggression_level, 2),
                "innovation_bias": round(agent.innovation_bias, 2),
                "capital_discipline": round(agent.capital_discipline, 2),
                "regulatory_sensitivity": round(agent.regulatory_sensitivity, 2),
                "time_horizon": agent.time_horizon,
                "memory_depth": agent.memory_depth,
            }
        )
    return rows


def strategic_pressure_summary(forces: MarketForces) -> dict[str, float]:
    return {
        "disruption_pressure": round(
            (
                forces.technology_shift_intensity
                + forces.threat_of_new_entry
                + forces.threat_of_substitution
            )
            / 3,
            3,
        ),
        "commercial_headwind": round(
            (forces.rivalry_intensity + forces.buyer_power + forces.macroeconomic_pressure) / 3,
            3,
        ),
        "resource_flexibility": round(
            (forces.capital_availability + forces.talent_availability + forces.customer_budget_index) / 3,
            3,
        ),
        "regulatory_heat": round(
            (forces.regulatory_pressure + forces.rivalry_intensity) / 2,
            3,
        ),
    }


def _pearson(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    numerator = sum(left * right for left, right in zip(centered_a, centered_b))
    denominator_left = math.sqrt(sum(value * value for value in centered_a))
    denominator_right = math.sqrt(sum(value * value for value in centered_b))
    denominator = denominator_left * denominator_right
    if denominator == 0:
        return 0.0
    return numerator / denominator


def company_metric_correlation_rows(world: WorldState) -> list[dict[str, float | str]]:
    metric_vectors = {
        "market_share": [company.market_share for company in world.companies.values()],
        "ai_capability": [company.ai_capability for company in world.companies.values()],
        "brand_trust": [company.brand_trust for company in world.companies.values()],
        "technical_debt": [company.technical_debt for company in world.companies.values()],
        "strategic_momentum": [company.strategic_momentum for company in world.companies.values()],
        "margin": [company.margin for company in world.companies.values()],
    }
    metric_names = list(metric_vectors)
    rows: list[dict[str, float | str]] = []
    for base_metric in metric_names:
        row: dict[str, float | str] = {"metric": base_metric}
        for compare_metric in metric_names:
            row[compare_metric] = round(
                _pearson(metric_vectors[base_metric], metric_vectors[compare_metric]),
                3,
            )
        rows.append(row)
    return rows
