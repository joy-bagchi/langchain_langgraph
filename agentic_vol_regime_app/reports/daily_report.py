"""Markdown daily report renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_vol_regime_app.contracts import (
    AlertRecord,
    BeliefRecord,
    CriticReviewRecord,
    FeatureRecord,
    HMMBeliefRecord,
    PolicyRecommendationRecord,
    TransitionProbabilityRecord,
)
from agentic_vol_regime_app.pomdp.states import REGIME_LABELS


def _format_probability(value: float) -> str:
    return f"{value:.2f}"


def _top_regime(belief_record: BeliefRecord) -> str:
    regime = max(belief_record.beliefs, key=belief_record.beliefs.get)
    return REGIME_LABELS.get(regime, regime)


def _slugify_report_model_name(model_name: str | None) -> str:
    if not model_name:
        return "unknown_model"
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(model_name))
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact or "unknown_model"


def render_daily_markdown(
    *,
    feature_record: FeatureRecord,
    belief_record: BeliefRecord,
    transition_record: TransitionProbabilityRecord,
    hmm_record: HMMBeliefRecord | None,
    alert_record: AlertRecord,
    policy_record: PolicyRecommendationRecord,
    critic_record: CriticReviewRecord,
    review_decision: dict[str, Any] | None = None,
    comparison_panel: list[dict[str, Any]] | None = None,
    hmm_variant_comparison: list[dict[str, Any]] | None = None,
    report_model_name: str | None = None,
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
    comparison_rows = ""
    if comparison_panel:
        comparison_rows = "\n".join(
            f"| {row['engine']} | {row['top_regime']} | {float(row['confidence']):.2f} | {row['recommended_posture']} |"
            for row in comparison_panel
        )
    overwrite_plan = "- None"
    if policy_record.overwrite_call_strike is not None and policy_record.overwrite_dte is not None:
        overwrite_plan = (
            f"- Suggested call overwrite strike: `{policy_record.overwrite_call_strike:.2f}`\n"
            f"- Suggested DTE: `{policy_record.overwrite_dte}`\n"
            f"- Notes: {policy_record.overwrite_rationale or 'Derived from current SPY spot and regime posture.'}"
        )

    hmm_section = "HMM advisory is unavailable for this run."
    if hmm_record is not None:
        hmm_prob_rows = "\n".join(
            f"| {state} | {_format_probability(probability)} |"
            for state, probability in hmm_record.state_probabilities.items()
        )
        emission_rows = "\n".join(
            f"| {state} | {_format_probability(probability)} | {_format_probability(hmm_record.persistence_lift.get(state, 0.0))} |"
            for state, probability in hmm_record.emission_state_probabilities.items()
        )
        state_summary_rows = "\n".join(
            (
                f"| {state} | "
                f"{float(summary.get('vix', 0.0)):.2f} | "
                f"{float(summary.get('realized_vol_21d', 0.0)):.2f} | "
                f"{float(summary.get('drawdown_21d', 0.0)):.2f} | "
                f"{float(summary.get('term_structure_slope', 0.0)):.2f} | "
                f"{float(summary.get('trend_persistence_21d', 0.0)):.2f} | "
                f"{float(summary.get('vvix_vix_ratio', 0.0)):.2f} |"
            )
            for state, summary in hmm_record.state_feature_summaries.items()
        )
        interpretation_lines = "\n".join(f"- {item}" for item in hmm_record.interpretation_notes) or "- None"
        hmm_transition_rows = "\n".join(
            "| " + " | ".join(f"{value:.2f}" for value in row) + " |"
            for row in hmm_record.transition_matrix
        ) or "| 0.00 | 0.00 | 0.00 | 0.00 |"
        hmm_warning_lines = "\n".join(f"- {item}" for item in hmm_record.warnings) or "- None"
        usage_lines = "\n".join(
            f"- {state}: {count}"
            for state, count in hmm_record.state_usage_counts.items()
        ) or "- None"
        sector_metric_lines = "\n".join(
            f"- {name}: `{value:.2f}`"
            for name, value in hmm_record.sector_metrics.items()
        ) or "- None"
        training_status_line = (
            "Trained and active."
            if hmm_record.is_trained
            else "Not trained enough for this run. HMM-specific regime inference is unavailable."
        )
        hmm_section = f"""## HMM Regime Persistence

Training status: `{hmm_record.training_status}`

Variant: `{hmm_record.variant_label}`

Model converged: `{hmm_record.model_converged}`

{training_status_line}

Top HMM state: `{hmm_record.top_state}`

Current-state expected duration: `{hmm_record.current_state_expected_duration_days:.2f}` days

### HMM State Probabilities

| State | Probability |
|---|---:|
{hmm_prob_rows}

### Emission vs Persistence

Emission-only top state: `{hmm_record.emission_top_state}`

| State | Emission-Only Probability | Persistence Lift |
|---|---:|---:|
{emission_rows}

### State Summaries

| State | Avg VIX | Avg RV21 | Avg Drawdown | Term Slope | Trend Persistence | VVIX/VIX |
|---|---:|---:|---:|---:|---:|---:|
{state_summary_rows}

### Interpretation Notes

{interpretation_lines}

### HMM Persistence

- Current state persists 5d: `{hmm_record.persistence_probabilities.get("current_state_5d", 0.0):.2f}`
- Current state persists 10d: `{hmm_record.persistence_probabilities.get("current_state_10d", 0.0):.2f}`
- Current state persists 21d: `{hmm_record.persistence_probabilities.get("current_state_21d", 0.0):.2f}`
- VOL_EXPANSION or HIGH_VOL within 5d: `{hmm_record.transition_probabilities.get("to_vol_expansion_or_high_vol_5d", 0.0):.2f}`
- VOL_EXPANSION or HIGH_VOL within 10d: `{hmm_record.transition_probabilities.get("to_vol_expansion_or_high_vol_10d", 0.0):.2f}`
- VOL_EXPANSION or HIGH_VOL within 21d: `{hmm_record.transition_probabilities.get("to_vol_expansion_or_high_vol_21d", 0.0):.2f}`

### HMM Transition Matrix

{hmm_transition_rows}

### State Usage Counts

{usage_lines}

### Sector Metrics

{sector_metric_lines}

Warnings:
{hmm_warning_lines}
"""

    comparison_section = ""
    if comparison_rows:
        comparison_section = f"""
## Belief Reconciliation

| Engine | Top Regime | Confidence | Recommended Posture |
|---|---|---:|---|
{comparison_rows}
"""

    model_variant_section = ""
    if hmm_variant_comparison:
        variant_rows = "\n".join(
            (
                f"| {row['model']} | {row['top_state']} | {float(row['confidence']):.2f} | "
                f"{float(row['expected_duration_days']):.2f} | {float(row['high_vol_transition_prob_10d']):.2f} | {row['recommendation']} |"
            )
            for row in hmm_variant_comparison
        )
        first_sector_metrics = next(
            (dict(row.get("sector_metrics", {})) for row in hmm_variant_comparison if dict(row.get("sector_metrics", {}))),
            {},
        )
        sector_section = ""
        if first_sector_metrics:
            interpretation = (
                "Low avg correlation plus low first eigenvalue share suggests sector independence; "
                "rising values suggest market-mode dominance and higher vol-expansion risk."
            )
            sector_section = f"""
### Sector Correlation / Market Mode

- `avg_pairwise_corr_21d`: `{float(first_sector_metrics.get('avg_pairwise_corr_21d', 0.0)):.2f}`
- `first_eigenvalue_share_21d`: `{float(first_sector_metrics.get('first_eigenvalue_share_21d', 0.0)):.2f}`
- Interpretation: {interpretation}
"""
        model_variant_section = f"""
## Model Variant Comparison

| Model | Top State | Confidence | Expected Duration | 10d High-Vol Transition Prob | Recommendation |
|---|---|---:|---:|---:|---|
{variant_rows}

{sector_section}
"""

    model_lines: list[str] = []
    if report_model_name:
        model_lines.append(f"Report model: `{report_model_name}`")
    report_model_version = hmm_record.model_version if hmm_record is not None else belief_record.model_version
    model_lines.append(f"Report model version: `{report_model_version}`")
    model_tag_block = "\n".join(model_lines)
    if model_tag_block:
        model_tag_block = f"\n{model_tag_block}"

    return f"""# Daily Volatility Regime Report

Date: {belief_record.as_of}
{model_tag_block}

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
| VIX term structure ({features.get("term_structure_symbol", "VIX3M")}) | {features.get("term_structure_state", "flat")} | Front/back of curve |
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

Overwrite implementation:
{overwrite_plan}

{comparison_section}

{model_variant_section}

{hmm_section}

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


def write_daily_report(
    markdown: str,
    *,
    report_root: Path,
    as_of: str,
    report_model_name: str | None = None,
) -> Path:
    """Persist the report to disk."""
    reports_dir = report_root / "daily"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_slug = _slugify_report_model_name(report_model_name)
    report_name = f"daily_regime_report_{as_of[:10]}_{model_slug}.md"
    report_path = reports_dir / report_name
    report_path.write_text(markdown, encoding="utf-8")
    return report_path
