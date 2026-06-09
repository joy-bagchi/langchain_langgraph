"""Deterministic single-turn simulation engine for the strategy game app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_strategy_game_app.contracts import StrategicAction, TurnResult, WorldState


SUPPORTED_ACTIONS = {
    "cut_price",
    "raise_price",
    "increase_rnd",
    "increase_marketing",
    "improve_operations",
    "expand_sales_capacity",
    "form_partnership",
    "lobby_regulator",
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _quarter_label(turn_number: int) -> str:
    quarter_index = (turn_number % 4) + 1
    return f"Q{quarter_index}"


def _rebalance_market_share(world: WorldState, actor_id: str, delta: float) -> None:
    actor = world.companies[actor_id]
    actor.market_share = _clamp(actor.market_share + delta, 0.0, 1.0)
    competitors = [company for company_id, company in world.companies.items() if company_id != actor_id]
    if not competitors:
        return
    if delta > 0:
        total_competitor_share = sum(company.market_share for company in competitors)
        if total_competitor_share <= 0:
            return
        for company in competitors:
            pull = delta * (company.market_share / total_competitor_share)
            company.market_share = _clamp(company.market_share - pull, 0.0, 1.0)
    elif delta < 0:
        redistribute = abs(delta)
        total_competitor_share = sum(company.market_share for company in competitors)
        if total_competitor_share <= 0:
            even_share = redistribute / len(competitors)
            for company in competitors:
                company.market_share = _clamp(company.market_share + even_share, 0.0, 1.0)
        else:
            for company in competitors:
                gain = redistribute * (company.market_share / total_competitor_share)
                company.market_share = _clamp(company.market_share + gain, 0.0, 1.0)


@dataclass(slots=True)
class StrategyGameEngine:
    """Deterministic single-turn engine where actions are validated before application."""

    max_actions_per_turn: int = 2

    def validate_actions(self, world: WorldState, actions: list[StrategicAction]) -> tuple[list[StrategicAction], list[dict[str, Any]]]:
        accepted: list[StrategicAction] = []
        rejected: list[dict[str, Any]] = []
        for index, action in enumerate(actions):
            if index >= self.max_actions_per_turn:
                rejected.append(
                    {
                        "action": action.to_dict(),
                        "reason": f"Exceeded max_actions_per_turn={self.max_actions_per_turn}.",
                    }
                )
                continue
            if action.action_type not in SUPPORTED_ACTIONS:
                rejected.append({"action": action.to_dict(), "reason": f"Unsupported action '{action.action_type}'."})
                continue
            actor = world.companies.get(action.actor_id)
            if actor is None:
                rejected.append({"action": action.to_dict(), "reason": f"Unknown actor '{action.actor_id}'."})
                continue
            if action.resource_cost > actor.cash:
                rejected.append({"action": action.to_dict(), "reason": "Insufficient cash for proposed action."})
                continue
            accepted.append(action)
        return accepted, rejected

    def run_turn(
        self,
        world: WorldState,
        actions: list[StrategicAction],
        *,
        advance_time: bool = True,
    ) -> tuple[WorldState, TurnResult]:
        state_before = world.to_dict()
        next_world = WorldState.from_dict(state_before)
        accepted_actions, rejected_actions = self.validate_actions(next_world, actions)
        forces = next_world.market_forces

        base_company_metrics = {
            company_id: {
                "revenue": company.revenue,
                "profit": company.profit,
                "cash": company.cash,
                "market_share": company.market_share,
                "strategic_momentum": company.strategic_momentum,
                "ai_capability": company.ai_capability,
                "brand_trust": company.brand_trust,
            }
            for company_id, company in next_world.companies.items()
        }

        financial_results: dict[str, Any] = {}
        market_results: dict[str, Any] = {}
        triggered_events: list[dict[str, Any]] = []

        passive_dynamics = self._apply_passive_dynamics(next_world)
        if passive_dynamics:
            market_results["passive_dynamics"] = passive_dynamics

        for action in accepted_actions:
            company = next_world.companies[action.actor_id]
            company.cash -= action.resource_cost
            action_metrics: dict[str, float] = {"resource_cost": action.resource_cost}

            if action.action_type == "cut_price":
                share_delta = 0.008 + (0.018 * action.intensity * ((forces.buyer_power + forces.rivalry_intensity) / 2))
                _rebalance_market_share(next_world, action.actor_id, share_delta)
                company.margin = _clamp(company.margin - (0.025 * action.intensity), -1.0, 1.0)
                company.revenue += company.revenue * (share_delta * max(0.45, forces.customer_budget_index))
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.015 * action.intensity), 0.0, 1.0)
                action_metrics.update({"market_share_delta": share_delta, "margin_delta": -(0.025 * action.intensity)})

            elif action.action_type == "raise_price":
                share_delta = -(0.006 + (0.012 * action.intensity * forces.buyer_power))
                _rebalance_market_share(next_world, action.actor_id, share_delta)
                company.margin = _clamp(company.margin + (0.02 * action.intensity), -1.0, 1.0)
                company.revenue += company.revenue * ((0.012 * action.intensity * max(0.4, company.brand_trust)) + share_delta)
                company.brand_trust = _clamp(company.brand_trust - (0.004 * action.intensity * forces.buyer_power), 0.0, 1.0)
                action_metrics.update({"market_share_delta": share_delta, "margin_delta": 0.02 * action.intensity})

            elif action.action_type == "increase_rnd":
                ai_gain = 0.045 * action.intensity * max(0.5, forces.technology_shift_intensity)
                quality_gain = 0.028 * action.intensity
                company.ai_capability = _clamp(company.ai_capability + ai_gain, 0.0, 1.0)
                company.product_quality = _clamp(company.product_quality + quality_gain, 0.0, 1.0)
                company.technical_debt = _clamp(company.technical_debt - (0.018 * action.intensity), 0.0, 1.0)
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.03 * action.intensity), 0.0, 1.0)
                action_metrics.update({"ai_capability_delta": ai_gain, "product_quality_delta": quality_gain})

            elif action.action_type == "increase_marketing":
                share_delta = 0.006 + (0.014 * action.intensity * max(0.45, forces.customer_budget_index))
                _rebalance_market_share(next_world, action.actor_id, share_delta)
                company.brand_trust = _clamp(company.brand_trust + (0.022 * action.intensity), 0.0, 1.0)
                company.customer_count += int(round(80 * action.intensity * max(0.4, forces.customer_budget_index)))
                company.revenue += company.revenue * (0.02 * action.intensity * max(0.45, forces.customer_budget_index))
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.025 * action.intensity), 0.0, 1.0)
                action_metrics.update({"market_share_delta": share_delta, "brand_trust_delta": 0.022 * action.intensity})

            elif action.action_type == "improve_operations":
                company.operational_efficiency = _clamp(company.operational_efficiency + (0.04 * action.intensity), 0.0, 1.0)
                company.margin = _clamp(company.margin + (0.017 * action.intensity), -1.0, 1.0)
                company.technical_debt = _clamp(company.technical_debt - (0.028 * action.intensity), 0.0, 1.0)
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.018 * action.intensity), 0.0, 1.0)
                action_metrics.update({"efficiency_delta": 0.04 * action.intensity, "margin_delta": 0.017 * action.intensity})

            elif action.action_type == "expand_sales_capacity":
                share_delta = 0.005 + (0.012 * action.intensity * max(0.35, forces.talent_availability))
                _rebalance_market_share(next_world, action.actor_id, share_delta)
                company.sales_capacity = _clamp(company.sales_capacity + (0.045 * action.intensity), 0.0, 1.0)
                company.customer_count += int(round(60 * action.intensity * max(0.35, forces.talent_availability)))
                company.revenue += company.revenue * (0.018 * action.intensity)
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.02 * action.intensity), 0.0, 1.0)
                action_metrics.update({"market_share_delta": share_delta, "sales_capacity_delta": 0.045 * action.intensity})

            elif action.action_type == "form_partnership":
                company.ai_capability = _clamp(company.ai_capability + (0.02 * action.intensity), 0.0, 1.0)
                company.sales_capacity = _clamp(company.sales_capacity + (0.025 * action.intensity), 0.0, 1.0)
                company.brand_trust = _clamp(company.brand_trust + (0.012 * action.intensity), 0.0, 1.0)
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.035 * action.intensity), 0.0, 1.0)
                company.revenue += company.revenue * (0.01 * action.intensity)
                action_metrics.update({"partnership_leverage": 0.025 * action.intensity})

            elif action.action_type == "lobby_regulator":
                company.regulatory_risk = _clamp(company.regulatory_risk - (0.03 * action.intensity), 0.0, 1.0)
                company.strategic_momentum = _clamp(company.strategic_momentum + (0.008 * action.intensity), 0.0, 1.0)
                next_world.market_forces.regulatory_pressure = _clamp(
                    next_world.market_forces.regulatory_pressure + (0.01 * action.intensity),
                    0.0,
                    1.0,
                )
                triggered_events.append(
                    {
                        "event_type": "policy_signal",
                        "title": f"{company.name} increased policy activity",
                        "severity": round(0.2 + (0.25 * action.intensity), 3),
                    }
                )
                action_metrics.update({"regulatory_risk_delta": -(0.03 * action.intensity)})

            financial_results[action.action_id] = action_metrics

        agent_payoffs: dict[str, Any] = {}
        for company_id, company in next_world.companies.items():
            prior = base_company_metrics[company_id]
            company.profit = company.revenue * company.margin
            company.cash += company.profit - prior["profit"]
            share_delta = company.market_share - prior["market_share"]
            profit_delta = company.profit - prior["profit"]
            momentum_delta = company.strategic_momentum - prior["strategic_momentum"]
            ai_delta = company.ai_capability - prior["ai_capability"]
            trust_delta = company.brand_trust - prior["brand_trust"]
            market_results[company_id] = {
                "market_share_delta": round(share_delta, 5),
                "ai_capability_delta": round(ai_delta, 5),
                "brand_trust_delta": round(trust_delta, 5),
            }
            agent_payoffs[company_id] = round(
                (share_delta * 8.0)
                + ((profit_delta / max(abs(prior["profit"]), 1.0)) * 0.9)
                + (momentum_delta * 2.0),
                4,
            )

        if advance_time:
            next_world.current_turn += 1
            next_world.current_period_label = _quarter_label(next_world.current_turn)

        turn_result = TurnResult(
            turn_number=next_world.current_turn if advance_time else world.current_turn,
            period_label=state_before["current_period_label"],
            actions_submitted=[action.to_dict() for action in actions],
            actions_accepted=[action.to_dict() for action in accepted_actions],
            actions_rejected=list(rejected_actions),
            state_before=state_before,
            state_after=next_world.to_dict(),
            financial_results=financial_results,
            market_results=market_results,
            agent_payoffs=agent_payoffs,
            triggered_events=triggered_events,
            pending_effects_created=[],
            narrative_summary=self._build_narrative(
                next_world,
                accepted_actions,
                rejected_actions,
                advance_time=advance_time,
            ),
        )
        next_world.historical_results.append(turn_result)
        return next_world, turn_result

    def _apply_passive_dynamics(self, world: WorldState) -> dict[str, dict[str, float]]:
        forces = world.market_forces
        updates: dict[str, dict[str, float]] = {}
        companies = world.companies
        if not companies:
            return updates

        disruption_pressure = (forces.technology_shift_intensity + forces.threat_of_new_entry) / 2
        for company_id, company in companies.items():
            baseline_share = company.market_share
            ai_gap = company.ai_capability - 0.65
            debt_drag = company.technical_debt - 0.4
            momentum_shift = (ai_gap * 0.035) - (debt_drag * 0.03) - (forces.macroeconomic_pressure * 0.01)
            if company.metadata.get("archetype") == "challenger":
                momentum_shift += 0.012 * disruption_pressure
            if company.metadata.get("archetype") == "incumbent":
                momentum_shift -= 0.008 * disruption_pressure
            company.strategic_momentum = _clamp(company.strategic_momentum + momentum_shift, 0.0, 1.0)

            passive_share_delta = (
                (company.ai_capability - company.technical_debt) * 0.006
                + (company.brand_trust - 0.5) * 0.0025
                + momentum_shift * 0.01
            )
            if company.metadata.get("archetype") == "challenger":
                passive_share_delta += 0.0035 * disruption_pressure
            if company.metadata.get("archetype") == "incumbent":
                passive_share_delta -= 0.0025 * disruption_pressure
            _rebalance_market_share(world, company_id, passive_share_delta)

            company.revenue += company.revenue * (
                forces.growth_rate * max(0.25, company.market_share) * 0.12
                + passive_share_delta * 0.8
            )
            company.margin = _clamp(
                company.margin
                - (forces.rivalry_intensity * 0.004)
                - (max(0.0, debt_drag) * 0.004)
                + (company.operational_efficiency - 0.5) * 0.003,
                -1.0,
                1.0,
            )
            company.regulatory_risk = _clamp(
                company.regulatory_risk + (forces.regulatory_pressure * 0.008) + (forces.rivalry_intensity * 0.003),
                0.0,
                1.0,
            )
            company.brand_trust = _clamp(
                company.brand_trust
                + (company.product_quality - 0.6) * 0.004
                - (forces.rivalry_intensity * 0.002),
                0.0,
                1.0,
            )
            updates[company_id] = {
                "market_share_delta": round(company.market_share - baseline_share, 5),
                "strategic_momentum_delta": round(momentum_shift, 5),
            }
        return updates

    def _build_narrative(
        self,
        world: WorldState,
        accepted_actions: list[StrategicAction],
        rejected_actions: list[dict[str, Any]],
        *,
        advance_time: bool,
    ) -> str:
        if not accepted_actions and not rejected_actions:
            return (
                "No player actions were proposed this turn. The ecosystem still advanced, "
                "and underlying market pressures continued to reshape company positions."
            )
        accepted_summary = ", ".join(
            f"{world.companies[action.actor_id].name} -> {action.action_type.replace('_', ' ')}"
            for action in accepted_actions
        ) or "no accepted actions"
        if rejected_actions:
            rejected_summary = "; ".join(
                f"{item['action']['action_type']}: {item['reason']}" for item in rejected_actions
            )
            return f"Accepted actions: {accepted_summary}. Rejected actions: {rejected_summary}."
        if advance_time:
            return f"Accepted actions: {accepted_summary}."
        return f"Accepted in-quarter actions: {accepted_summary}. The simulation clock remained inside the current quarter."
