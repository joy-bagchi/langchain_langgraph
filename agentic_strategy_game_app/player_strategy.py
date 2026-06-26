"""Deterministic player-strategy interpretation for the strategy game app."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from agentic_strategy_game_app.contracts import StrategyInterpretation, StrategicAction, UserStrategyIntent, WorldState


@dataclass(frozen=True, slots=True)
class _ActionRule:
    action_type: str
    phrases: tuple[str, ...]
    default_intensity: float
    internal_systems: tuple[str, ...]
    external_systems: tuple[str, ...]
    goal: str
    rationale_template: str
    required_tokens: tuple[str, ...] = ()
    target_kind: str = "internal"


ACTION_RULES: tuple[_ActionRule, ...] = (
    _ActionRule(
        action_type="cut_price",
        phrases=("cut price", "lower price", "discount", "price cut", "cheaper"),
        default_intensity=0.48,
        internal_systems=("pricing", "margin management"),
        external_systems=("customers", "competitors"),
        goal="win share through sharper pricing",
        rationale_template="The strategy emphasizes pricing aggression to move demand quickly.",
    ),
    _ActionRule(
        action_type="raise_price",
        phrases=("raise price", "increase price", "premium pricing", "take price"),
        default_intensity=0.34,
        internal_systems=("pricing", "gross margin"),
        external_systems=("customers", "capital markets"),
        goal="protect monetization and pricing power",
        rationale_template="The strategy emphasizes monetization discipline over aggressive share capture.",
    ),
    _ActionRule(
        action_type="increase_rnd",
        phrases=("r&d", "research", "invest in ai", "build ai", "launch ai", "product investment", "increase rnd"),
        required_tokens=("ai",),
        default_intensity=0.67,
        internal_systems=("product", "engineering", "innovation budget"),
        external_systems=("customers", "competitors", "talent market"),
        goal="accelerate product and AI capability",
        rationale_template="The strategy prioritizes product velocity and differentiated capability.",
    ),
    _ActionRule(
        action_type="increase_marketing",
        phrases=("marketing", "campaign", "brand", "demand gen", "go to market", "promote"),
        default_intensity=0.56,
        internal_systems=("marketing", "pipeline generation"),
        external_systems=("customers", "media", "competitors"),
        goal="expand awareness and demand capture",
        rationale_template="The strategy pushes awareness and category capture through go-to-market spend.",
    ),
    _ActionRule(
        action_type="improve_operations",
        phrases=("operations", "efficiency", "process", "cost discipline", "streamline", "productivity"),
        default_intensity=0.52,
        internal_systems=("operations", "cost base", "delivery"),
        external_systems=("customers", "capital markets"),
        goal="improve execution efficiency and resilience",
        rationale_template="The strategy seeks operating leverage and cleaner execution.",
    ),
    _ActionRule(
        action_type="expand_sales_capacity",
        phrases=("expand sales", "sales team", "hire reps", "distribution", "field sales"),
        default_intensity=0.58,
        internal_systems=("sales", "coverage", "hiring"),
        external_systems=("customers", "labor market"),
        goal="increase distribution and selling capacity",
        rationale_template="The strategy leans on distribution expansion to convert demand into revenue.",
    ),
    _ActionRule(
        action_type="form_partnership",
        phrases=("partnership", "partner", "alliance", "hyperscaler", "channel partner"),
        default_intensity=0.44,
        internal_systems=("business development", "platform strategy"),
        external_systems=("suppliers", "platform partners", "customers"),
        goal="improve ecosystem leverage through partnerships",
        rationale_template="The strategy uses ecosystem leverage instead of only internal build capacity.",
        target_kind="external",
    ),
    _ActionRule(
        action_type="lobby_regulator",
        phrases=("lobby", "regulator", "policy outreach", "shape regulation"),
        default_intensity=0.38,
        internal_systems=("legal", "policy"),
        external_systems=("regulator", "media"),
        goal="shape the regulatory environment",
        rationale_template="The strategy explicitly tries to influence external rule-setting.",
        target_kind="external",
    ),
)


def _detect_intensity(raw_text: str, default: float) -> float:
    text = raw_text.lower()
    if any(token in text for token in ("aggressive", "aggressively", "heavily", "hard", "rapidly", "strongly")):
        return min(0.9, default + 0.25)
    if any(token in text for token in ("slight", "slightly", "careful", "carefully", "light", "modest", "modestly")):
        return max(0.2, default - 0.2)
    if any(token in text for token in ("moderate", "moderately", "steady", "steadily", "balanced")):
        return default
    return default


def _estimate_resource_cost(action_type: str, intensity: float, company_revenue: float, company_cash: float) -> float:
    revenue = max(company_revenue, 0.0)
    cash = max(company_cash, 0.0)
    if action_type == "increase_rnd":
        return round(revenue * 0.05 * intensity, 2)
    if action_type == "increase_marketing":
        return round(revenue * 0.03 * intensity, 2)
    if action_type == "improve_operations":
        return round(revenue * 0.015 * intensity, 2)
    if action_type == "expand_sales_capacity":
        return round(revenue * 0.025 * intensity, 2)
    if action_type == "form_partnership":
        return round(revenue * 0.01 * intensity, 2)
    if action_type == "lobby_regulator":
        return round(revenue * 0.006 * intensity, 2)
    if action_type == "cut_price":
        return round(revenue * 0.01 * intensity, 2)
    if action_type == "raise_price":
        return 0.0
    return round(cash * 0.005 * intensity, 2)


def interpret_player_strategy(
    *,
    world: WorldState,
    actor_id: str,
    raw_strategy: str,
    max_actions: int = 2,
) -> StrategyInterpretation:
    """Translate plain-English player strategy into structured action proposals."""
    actor = world.companies.get(actor_id)
    if actor is None:
        raise ValueError(f"Player actor '{actor_id}' is not a controllable company in this scenario.")
    intent = UserStrategyIntent(
        intent_id=str(uuid4()),
        actor_id=actor_id,
        raw_strategy=raw_strategy,
    )
    normalized = raw_strategy.lower()
    matched: list[tuple[int, _ActionRule]] = []
    warnings: list[str] = []
    internal_systems: list[str] = []
    external_systems: list[str] = []
    interpreted_goals: list[str] = []

    for rule in ACTION_RULES:
        positions = [normalized.find(phrase) for phrase in rule.phrases if normalized.find(phrase) >= 0]
        if not positions and rule.required_tokens and all(token in normalized for token in rule.required_tokens):
            positions = [normalized.find(token) for token in rule.required_tokens if normalized.find(token) >= 0]
        if positions:
            matched.append((min(positions), rule))

    matched.sort(key=lambda item: item[0])

    if not matched:
        warnings.append(
            "No supported strategic actions were recognized. Try phrases about pricing, AI/R&D, marketing, operations, sales capacity, partnerships, or regulators."
        )
        return StrategyInterpretation(
            actor_id=actor.company_id,
            actor_name=actor.name,
            raw_strategy=intent.raw_strategy,
            summary=f"No structured actions could be inferred for {actor.name}.",
            actions=[],
            internal_systems=[],
            external_systems=[],
            interpreted_goals=[],
            warnings=warnings,
            metadata={"intent_id": intent.intent_id, "max_actions": max_actions},
        )

    deduped_rules: list[_ActionRule] = []
    seen_action_types: set[str] = set()
    for _, rule in matched:
        if rule.action_type in seen_action_types:
            continue
        deduped_rules.append(rule)
        seen_action_types.add(rule.action_type)

    if len(deduped_rules) > max_actions:
        warnings.append(
            f"Strategy mentioned {len(deduped_rules)} recognizable actions, but only the first {max_actions} are kept for the current turn."
        )
    selected_rules = deduped_rules[:max_actions]

    actions: list[StrategicAction] = []
    for index, rule in enumerate(selected_rules, start=1):
        intensity = _detect_intensity(normalized, rule.default_intensity)
        resource_cost = _estimate_resource_cost(rule.action_type, intensity, actor.revenue, actor.cash)
        rationale = f"{rule.rationale_template} Requested by player strategy for {actor.name}."
        actions.append(
            StrategicAction(
                action_id=f"{actor_id}-{rule.action_type}-{index}",
                actor_id=actor_id,
                action_type=rule.action_type,
                intensity=round(intensity, 2),
                resource_cost=resource_cost,
                expected_effects={"goal": rule.goal, "target_kind": rule.target_kind},
                constraints=[],
                rationale=rationale,
                metadata={"source": "player_strategy_interpreter"},
            )
        )
        for system in rule.internal_systems:
            if system not in internal_systems:
                internal_systems.append(system)
        for system in rule.external_systems:
            if system not in external_systems:
                external_systems.append(system)
        if rule.goal not in interpreted_goals:
            interpreted_goals.append(rule.goal)

    summary = (
        f"Interpreted player strategy for {actor.name}: "
        + ", ".join(action.action_type.replace("_", " ") for action in actions)
        + "."
    )
    return StrategyInterpretation(
        actor_id=actor.company_id,
        actor_name=actor.name,
        raw_strategy=intent.raw_strategy,
        summary=summary,
        actions=actions,
        internal_systems=internal_systems,
        external_systems=external_systems,
        interpreted_goals=interpreted_goals,
        warnings=warnings,
        metadata={"intent_id": intent.intent_id, "max_actions": max_actions},
    )
