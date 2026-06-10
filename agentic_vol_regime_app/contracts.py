"""Typed contracts for the volatility regime app."""

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


@dataclass(slots=True)
class OptionGreekRecord:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_vol: float | None = None
    opt_price: float | None = None
    pv_dividend: float | None = None
    und_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class OptionQuoteRecord:
    symbol: str
    expiry: str
    strike: float
    right: str
    exchange: str
    currency: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    mark: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_size: float | None = None
    multiplier: str | None = None
    greeks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class OptionChainSnapshot:
    underlying_symbol: str
    underlying_price: float | None
    fetched_at: str
    exchange: str
    currency: str
    expirations: list[str] = field(default_factory=list)
    strikes: list[float] = field(default_factory=list)
    rights: list[str] = field(default_factory=list)
    option_quotes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class ObservationRecord:
    schema_version: str
    as_of: str
    source: str
    symbols: dict[str, dict[str, Any]]
    history: dict[str, list[float]] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    option_chain: dict[str, Any] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class FeatureRecord:
    schema_version: str
    as_of: str
    feature_set_version: str
    features: dict[str, Any]
    missing_features: list[str]
    lookback_windows: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class BeliefRecord:
    schema_version: str
    as_of: str
    model_version: str
    beliefs: dict[str, float]
    belief_delta: dict[str, float]
    entropy: float
    confidence: float
    drivers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class TransitionProbabilityRecord:
    schema_version: str
    as_of: str
    model_version: str
    transition_probabilities: dict[str, float]
    top_predictive_factors: list[str]
    confirming_features_count: int

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class HMMBeliefRecord:
    schema_version: str
    model_name: str
    model_version: str
    as_of: str
    is_trained: bool
    training_status: str
    state_probabilities: dict[str, float]
    top_state: str
    transition_matrix: list[list[float]]
    expected_duration_days: dict[str, float]
    current_state_expected_duration_days: float
    persistence_probabilities: dict[str, float]
    transition_probabilities: dict[str, float]
    confidence: float
    warnings: list[str]
    drivers: list[str]
    interpretation_notes: list[str] = field(default_factory=list)
    state_label_mapping: dict[str, str] = field(default_factory=dict)
    emission_state_probabilities: dict[str, float] = field(default_factory=dict)
    emission_top_state: str = ""
    persistence_lift: dict[str, float] = field(default_factory=dict)
    state_feature_summaries: dict[str, dict[str, float]] = field(default_factory=dict)
    training_row_count: int = 0
    configured_train_window: int = 0
    inference_feature_vector: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class AlertRecord:
    schema_version: str
    alert_id: str
    as_of: str
    severity: str
    alert_type: str
    headline: str
    probabilities: dict[str, float]
    belief_state: dict[str, float]
    drivers: list[str]
    recommended_review: list[str]
    requires_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class PolicyRecommendationRecord:
    schema_version: str
    as_of: str
    recommended_action: str
    confidence: float
    rationale: list[str]
    risk_notes: list[str]
    requires_human_review: bool
    overwrite_call_strike: float | None = None
    overwrite_dte: int | None = None
    overwrite_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)


@dataclass(slots=True)
class CriticReviewRecord:
    schema_version: str
    as_of: str
    verdict: str
    findings: list[str]
    requires_human_review: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)
