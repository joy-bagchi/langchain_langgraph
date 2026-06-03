"""Markdown daily report renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_vol_regime_app.contracts import (
    AlertRecord,
    BeliefRecord,
    CriticReviewRecord,
    FeatureRecord,
    PolicyRecommendationRecord,
    TransitionProbabilityRecord,
)
from agentic_vol_regime_app.pomdp.states import REGIME_LABELS


def _format_probability(value: float) -> str:
    return f"{value:.2f}"


def _top_regime(belief_record: BeliefRecord) -> str:
    regime = max(belief_record.beliefs, key=belief_record.beliefs.get)
    return REGIME_LABELS.get(regime, regime)


def render_daily_markdown(
    *,
    feature_record: FeatureRecord,
    belief_record: BeliefRecord,
    transition_record: TransitionProbabilityRecord,
    alert_record: AlertRecord,
    policy_record: PolicyRecommendationRecord,
    critic_record: CriticReviewRecord,
    review_decision: dict[str, Any] | None = None,
) -> str:
    """Render the user-facing daily markdown report."""
    alert_label = alert_record.severity
    top_regime = _top_regime(belief_record)
    review_required = "Yes" if (alert_record.requires_human_review or critic_record.requires_human_review) else "No"
    review_decision_line = "Not required"
    if review_decision:
        review_decision_line = str(review_decision.get("decision", "pending")).upper()

    features = feature_record.features
    belief_rows = "\n".join(
        f"| {REGIME_LABELS.get(regime, regime)} | {_format_probability(probability)} |"
        for regime, probability in belief_record.beliefs.items()
    )
    rationale = "\n".join(f"- {item}" for item in policy_record.rationale)
    risk_notes = "\n".join(f"- {item}" for item in policy_record.risk_notes) or "- None"
    drivers = "\n".join(f"- {item}" for item in alert_record.drivers) or "- None"
    critic_findings = "\n".join(f"- {item}" for item in critic_record.findings) or "- None"

    return f"""# Daily Volatility Regime Report

Date: {belief_record.as_of}

## Summary

Current regime belief favors: `{top_regime}`

Transition risk: `{alert_label}`

Recommended posture: `{policy_record.recommended_action}`

## Belief State

| Regime | Probability |
|---|---:|
{belief_rows}

## Key Signals

| Signal | Value | Interpretation |
|---|---:|---|
| VIX | {features.get("vix", 0.0):.2f} | Current implied-vol level |
| VVIX | {features.get("vvix", 0.0):.2f} | Vol-of-vol state |
| VVIX/VIX | {features.get("vvix_vix_ratio", 0.0):.2f} | Convexity stress ratio |
| VVIX/VIX z-score | {float(features.get("vvix_vix_z_22d", 0.0) or 0.0):.2f} | Relative convexity stress |
| VIX term structure | {features.get("term_structure_state", "flat")} | Front/back of curve |
| Realized vol trend | {float(features.get("realized_vol_acceleration", 0.0) or 0.0):.3f} | Short-vs-medium realized vol |

## Predictive Alerts

Severity: `{alert_record.severity}`

Headline: {alert_record.headline}

Drivers:
{drivers}

## Policy Recommendation

Recommended action: `{policy_record.recommended_action}`

Rationale:
{rationale}

Risk notes:
{risk_notes}

## Model Confidence

Confidence: {belief_record.confidence:.2f}

Uncertainty / entropy: {belief_record.entropy:.2f}

## Critic Review

Verdict: `{critic_record.verdict}`

Findings:
{critic_findings}

## Required Human Review

Required: {review_required}

Review decision: {review_decision_line}
"""


def write_daily_report(markdown: str, *, report_root: Path, as_of: str) -> Path:
    """Persist the report to disk."""
    reports_dir = report_root / "daily"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"daily_regime_report_{as_of[:10]}.md"
    report_path = reports_dir / report_name
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
