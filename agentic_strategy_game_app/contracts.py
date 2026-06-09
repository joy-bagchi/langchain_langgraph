"""Typed contracts for the agentic business strategy game."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _asdict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [_asdict(item) for item in value]
    if isinstance(value, dict):
        return {key: _asdict(item) for key, item in value.items()}
    return value


def _bounded(name: str, value: float, minimum: float = 0.0, maximum: float = 1.0) -> None:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}; received {value}.")


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")


@dataclass(slots=True)
class MarketForces:
    market_size: float
    growth_rate: float
    customer_budget_index: float
    rivalry_intensity: float
    buyer_power: float
    supplier_power: float
    threat_of_new_entry: float
    threat_of_substitution: float
    regulatory_pressure: float
    capital_availability: float
    talent_availability: float
    technology_shift_intensity: float
    macroeconomic_pressure: float

    def __post_init__(self) -> None:
        _non_negative("market_size", self.market_size)
        for name in (
            "customer_budget_index",
            "rivalry_intensity",
            "buyer_power",
            "supplier_power",
            "threat_of_new_entry",
            "threat_of_substitution",
            "regulatory_pressure",
            "capital_availability",
            "talent_availability",
            "technology_shift_intensity",
            "macroeconomic_pressure",
        ):
            _bounded(name, float(getattr(self, name)))
        if self.growth_rate < -1.0 or self.growth_rate > 3.0:
            raise ValueError(f"growth_rate must be between -1.0 and 3.0; received {self.growth_rate}.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketForces":
        return cls(**dict(payload))


@dataclass(slots=True)
class CompanyState:
    company_id: str
    name: str
    cash: float
    revenue: float
    margin: float
    profit: float
    market_share: float
    customer_count: int
    product_quality: float
    ai_capability: float
    brand_trust: float
    sales_capacity: float
    operational_efficiency: float
    talent_density: float
    technical_debt: float
    regulatory_risk: float
    strategic_momentum: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.company_id.strip():
            raise ValueError("company_id is required.")
        if not self.name.strip():
            raise ValueError("name is required.")
        for name in ("cash", "revenue", "customer_count", "sales_capacity"):
            _non_negative(name, float(getattr(self, name)))
        if self.margin < -1.0 or self.margin > 1.0:
            raise ValueError(f"margin must be between -1.0 and 1.0; received {self.margin}.")
        for name in (
            "market_share",
            "product_quality",
            "ai_capability",
            "brand_trust",
            "operational_efficiency",
            "talent_density",
            "technical_debt",
            "regulatory_risk",
            "strategic_momentum",
        ):
            _bounded(name, float(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompanyState":
        return cls(**dict(payload))


@dataclass(slots=True)
class AgentProfile:
    agent_id: str
    name: str
    agent_type: str
    objective_function: str
    risk_tolerance: float
    cooperation_tendency: float
    aggression_level: float
    price_sensitivity: float
    innovation_bias: float
    capital_discipline: float
    regulatory_sensitivity: float
    time_horizon: int
    memory_depth: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("agent_id", "name", "agent_type", "objective_function"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required.")
        for name in (
            "risk_tolerance",
            "cooperation_tendency",
            "aggression_level",
            "price_sensitivity",
            "innovation_bias",
            "capital_discipline",
            "regulatory_sensitivity",
        ):
            _bounded(name, float(getattr(self, name)))
        if self.time_horizon <= 0:
            raise ValueError("time_horizon must be positive.")
        if self.memory_depth < 0:
            raise ValueError("memory_depth must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentProfile":
        return cls(**dict(payload))


@dataclass(slots=True)
class AgentObservation:
    agent_id: str
    turn_number: int
    period_label: str
    visible_market_forces: dict[str, Any]
    visible_company_state: dict[str, Any]
    visible_competitor_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    known_events: list[dict[str, Any]] = field(default_factory=list)
    uncertain_beliefs: list[str] = field(default_factory=list)
    memory_summary: list[str] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    confidence_by_signal: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("agent_id is required.")
        if self.turn_number < 0:
            raise ValueError("turn_number must be non-negative.")
        if not self.period_label.strip():
            raise ValueError("period_label is required.")
        for signal, confidence in self.confidence_by_signal.items():
            _bounded(f"confidence_by_signal[{signal}]", float(confidence))

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentObservation":
        return cls(**dict(payload))


@dataclass(slots=True)
class StrategicAction:
    action_id: str
    actor_id: str
    action_type: str
    intensity: float
    resource_cost: float
    expected_effects: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    rationale: str = ""
    target_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("action_id", "actor_id", "action_type"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required.")
        _bounded("intensity", float(self.intensity))
        _non_negative("resource_cost", float(self.resource_cost))

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategicAction":
        return cls(**dict(payload))


@dataclass(slots=True)
class UserStrategyIntent:
    intent_id: str
    actor_id: str
    raw_strategy: str
    planning_horizon: str = "current_turn"
    interpreted_goals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.intent_id.strip():
            raise ValueError("intent_id is required.")
        if not self.actor_id.strip():
            raise ValueError("actor_id is required.")
        if not self.raw_strategy.strip():
            raise ValueError("raw_strategy is required.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserStrategyIntent":
        return cls(**dict(payload))


@dataclass(slots=True)
class StrategyInterpretation:
    actor_id: str
    actor_name: str
    raw_strategy: str
    summary: str
    actions: list[StrategicAction] = field(default_factory=list)
    internal_systems: list[str] = field(default_factory=list)
    external_systems: list[str] = field(default_factory=list)
    interpreted_goals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ValueError("actor_id is required.")
        if not self.actor_name.strip():
            raise ValueError("actor_name is required.")
        if not self.raw_strategy.strip():
            raise ValueError("raw_strategy is required.")
        if not self.summary.strip():
            raise ValueError("summary is required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "actor_name": self.actor_name,
            "raw_strategy": self.raw_strategy,
            "summary": self.summary,
            "actions": [action.to_dict() for action in self.actions],
            "internal_systems": list(self.internal_systems),
            "external_systems": list(self.external_systems),
            "interpreted_goals": list(self.interpreted_goals),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyInterpretation":
        return cls(
            actor_id=str(payload["actor_id"]),
            actor_name=str(payload["actor_name"]),
            raw_strategy=str(payload["raw_strategy"]),
            summary=str(payload["summary"]),
            actions=[StrategicAction.from_dict(dict(item)) for item in payload.get("actions", [])],
            internal_systems=list(payload.get("internal_systems", [])),
            external_systems=list(payload.get("external_systems", [])),
            interpreted_goals=list(payload.get("interpreted_goals", [])),
            warnings=list(payload.get("warnings", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class VCPitchTurn:
    speaker: str
    content: str
    round_number: int

    def __post_init__(self) -> None:
        if self.speaker not in {"player", "vc_agent"}:
            raise ValueError("speaker must be 'player' or 'vc_agent'.")
        if not self.content.strip():
            raise ValueError("content is required.")
        if self.round_number < 0:
            raise ValueError("round_number must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VCPitchTurn":
        return cls(**dict(payload))


@dataclass(slots=True)
class VCPitchDecision:
    outcome: str
    amount_offered: float
    equity_requested: float
    terms: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in {"decline", "fund", "counter_offer"}:
            raise ValueError("outcome must be decline, fund, or counter_offer.")
        _non_negative("amount_offered", float(self.amount_offered))
        _bounded("equity_requested", float(self.equity_requested))

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VCPitchDecision":
        return cls(**dict(payload))


@dataclass(slots=True)
class VCPitchAgentResponse:
    mode: str
    summary: str
    diligence_focus: list[str] = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)
    tentative_signal: str = "undecided"
    decision: VCPitchDecision | None = None
    raw_response: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"questioning", "decision"}:
            raise ValueError("mode must be questioning or decision.")
        if not self.summary.strip():
            raise ValueError("summary is required.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "diligence_focus": list(self.diligence_focus),
            "followup_questions": list(self.followup_questions),
            "tentative_signal": self.tentative_signal,
            "decision": self.decision.to_dict() if self.decision else None,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VCPitchAgentResponse":
        decision_payload = payload.get("decision")
        return cls(
            mode=str(payload["mode"]),
            summary=str(payload["summary"]),
            diligence_focus=list(payload.get("diligence_focus", [])),
            followup_questions=list(payload.get("followup_questions", [])),
            tentative_signal=str(payload.get("tentative_signal", "undecided")),
            decision=VCPitchDecision.from_dict(dict(decision_payload)) if decision_payload else None,
            raw_response=str(payload.get("raw_response", "")),
        )


@dataclass(slots=True)
class VCPitchSessionState:
    session_id: str
    actor_id: str
    company_name: str
    capital_requested: float
    equity_offered: float
    strategy_summary: str
    round_number: int = 0
    status: str = "active"
    transcript: list[VCPitchTurn] = field(default_factory=list)
    latest_agent_response: VCPitchAgentResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required.")
        if not self.actor_id.strip():
            raise ValueError("actor_id is required.")
        if not self.company_name.strip():
            raise ValueError("company_name is required.")
        if not self.strategy_summary.strip():
            raise ValueError("strategy_summary is required.")
        _non_negative("capital_requested", float(self.capital_requested))
        _bounded("equity_offered", float(self.equity_offered))
        if self.round_number < 0:
            raise ValueError("round_number must be non-negative.")
        if self.status not in {"active", "decision_made"}:
            raise ValueError("status must be active or decision_made.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "company_name": self.company_name,
            "capital_requested": self.capital_requested,
            "equity_offered": self.equity_offered,
            "strategy_summary": self.strategy_summary,
            "round_number": self.round_number,
            "status": self.status,
            "transcript": [turn.to_dict() for turn in self.transcript],
            "latest_agent_response": self.latest_agent_response.to_dict() if self.latest_agent_response else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VCPitchSessionState":
        latest_payload = payload.get("latest_agent_response")
        return cls(
            session_id=str(payload["session_id"]),
            actor_id=str(payload["actor_id"]),
            company_name=str(payload["company_name"]),
            capital_requested=float(payload["capital_requested"]),
            equity_offered=float(payload["equity_offered"]),
            strategy_summary=str(payload["strategy_summary"]),
            round_number=int(payload.get("round_number", 0)),
            status=str(payload.get("status", "active")),
            transcript=[VCPitchTurn.from_dict(dict(item)) for item in payload.get("transcript", [])],
            latest_agent_response=VCPitchAgentResponse.from_dict(dict(latest_payload)) if latest_payload else None,
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class PendingEffect:
    effect_id: str
    actor_id: str
    effect_type: str
    applies_on_turn: int
    payload: dict[str, Any]
    created_turn: int
    description: str = ""

    def __post_init__(self) -> None:
        for name in ("effect_id", "actor_id", "effect_type"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required.")
        if self.applies_on_turn < 1:
            raise ValueError("applies_on_turn must be at least 1.")
        if self.created_turn < 0:
            raise ValueError("created_turn must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PendingEffect":
        return cls(**dict(payload))


@dataclass(slots=True)
class GameEvent:
    event_id: str
    event_type: str
    title: str
    description: str
    severity: float
    turn_triggered: int
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_turns: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "title", "source"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required.")
        _bounded("severity", float(self.severity))
        if self.turn_triggered < 0:
            raise ValueError("turn_triggered must be non-negative.")
        if self.duration_turns < 0:
            raise ValueError("duration_turns must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameEvent":
        return cls(**dict(payload))


@dataclass(slots=True)
class TurnResult:
    turn_number: int
    period_label: str
    actions_submitted: list[dict[str, Any]]
    actions_accepted: list[dict[str, Any]]
    actions_rejected: list[dict[str, Any]]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    financial_results: dict[str, Any]
    market_results: dict[str, Any]
    agent_payoffs: dict[str, Any]
    triggered_events: list[dict[str, Any]] = field(default_factory=list)
    pending_effects_created: list[dict[str, Any]] = field(default_factory=list)
    narrative_summary: str = ""

    def __post_init__(self) -> None:
        if self.turn_number < 0:
            raise ValueError("turn_number must be non-negative.")
        if not self.period_label.strip():
            raise ValueError("period_label is required.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnResult":
        return cls(**dict(payload))


@dataclass(slots=True)
class SimulationConfig:
    scenario_name: str
    random_seed: int = 7
    turn_interval_label: str = "quarter"
    max_turns: int = 12
    uncertainty_enabled: bool = True
    allow_random_events: bool = True
    max_actions_per_turn: int = 2
    visibility_mode: str = "partial_information"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_name.strip():
            raise ValueError("scenario_name is required.")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        if self.max_actions_per_turn <= 0:
            raise ValueError("max_actions_per_turn must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimulationConfig":
        return cls(**dict(payload))


@dataclass(slots=True)
class WorldState:
    current_turn: int
    current_period_label: str
    scenario_name: str
    market_forces: MarketForces
    companies: dict[str, CompanyState]
    ecosystem_agents: dict[str, AgentProfile]
    active_events: list[GameEvent] = field(default_factory=list)
    historical_results: list[TurnResult] = field(default_factory=list)
    pending_effects: list[PendingEffect] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_turn < 0:
            raise ValueError("current_turn must be non-negative.")
        if not self.current_period_label.strip():
            raise ValueError("current_period_label is required.")
        if not self.scenario_name.strip():
            raise ValueError("scenario_name is required.")
        if not self.companies:
            raise ValueError("companies must not be empty.")
        if not self.ecosystem_agents:
            raise ValueError("ecosystem_agents must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_turn": self.current_turn,
            "current_period_label": self.current_period_label,
            "scenario_name": self.scenario_name,
            "market_forces": self.market_forces.to_dict(),
            "companies": {key: value.to_dict() for key, value in self.companies.items()},
            "ecosystem_agents": {key: value.to_dict() for key, value in self.ecosystem_agents.items()},
            "active_events": [value.to_dict() for value in self.active_events],
            "historical_results": [value.to_dict() for value in self.historical_results],
            "pending_effects": [value.to_dict() for value in self.pending_effects],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorldState":
        return cls(
            current_turn=int(payload["current_turn"]),
            current_period_label=str(payload["current_period_label"]),
            scenario_name=str(payload["scenario_name"]),
            market_forces=MarketForces.from_dict(dict(payload["market_forces"])),
            companies={
                key: CompanyState.from_dict(dict(value))
                for key, value in dict(payload["companies"]).items()
            },
            ecosystem_agents={
                key: AgentProfile.from_dict(dict(value))
                for key, value in dict(payload["ecosystem_agents"]).items()
            },
            active_events=[GameEvent.from_dict(dict(value)) for value in payload.get("active_events", [])],
            historical_results=[TurnResult.from_dict(dict(value)) for value in payload.get("historical_results", [])],
            pending_effects=[PendingEffect.from_dict(dict(value)) for value in payload.get("pending_effects", [])],
            metadata=dict(payload.get("metadata", {})),
        )
