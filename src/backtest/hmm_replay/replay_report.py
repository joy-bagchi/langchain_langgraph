from __future__ import annotations

from typing import Any

import pandas as pd


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"
    return frame.to_markdown(index=False)


def _model_usefulness_lines(economic_summary: pd.DataFrame) -> str:
    if economic_summary.empty:
        return "No economic summary rows were available."
    lines: list[str] = []
    for _, row in economic_summary.iterrows():
        model = str(row.get("model_name", "unknown"))
        horizon = int(row.get("horizon", 0))
        exact = float(row.get("accuracy", 0.0))
        adjacent = float(row.get("adjacent_tolerant_accuracy", 0.0))
        missed_risk = float(row.get("missed_risk_rate", 0.0))
        false_alarm = float(row.get("false_alarm_rate", 0.0))
        brier_spike = float(row.get("brier_vix_spike", 0.0))
        direction = float(row.get("combined_vol_directional_accuracy", 0.0))
        if adjacent > exact + 0.1:
            lines.append(
                f"- `{model}` T+{horizon}: exact accuracy is lower than adjacent-tolerant accuracy, "
                "so most misses are neighboring-regime errors."
            )
        else:
            lines.append(
                f"- `{model}` T+{horizon}: exact and adjacent accuracy are close, so label misses are less concentrated in neighbor states."
            )
        if missed_risk > 0.2:
            lines.append(
                f"- `{model}` T+{horizon}: missed-risk rate is elevated ({missed_risk:.1%}); risk transitions may be under-warned."
            )
        elif false_alarm > 0.25:
            lines.append(
                f"- `{model}` T+{horizon}: false-alarm rate is elevated ({false_alarm:.1%}); model may over-call expansion risk."
            )
        else:
            lines.append(
                f"- `{model}` T+{horizon}: false alarms ({false_alarm:.1%}) and missed risks ({missed_risk:.1%}) are balanced."
            )
        lines.append(
            f"- `{model}` T+{horizon}: Brier VIX Spike={brier_spike:.4f}, combined vol-direction accuracy={direction:.1%}."
        )
    return "\n".join(lines)


def _disagreement_interpretation(disagreement_summary: pd.DataFrame) -> str:
    if disagreement_summary.empty:
        return "No disagreement attribution rows were available."
    lines: list[str] = []
    downgrade_success = float(disagreement_summary["downgrade_success_rate"].mean())
    false_suppression = float(disagreement_summary["false_suppression_rate"].mean())
    upgrade_success = float(disagreement_summary["upgrade_success_rate"].mean())
    ties = float(disagreement_summary["tie_rate"].mean())
    if downgrade_success > 0.6:
        lines.append("- HMM v3 appears to be usefully suppressing false vol-expansion warnings.")
    if false_suppression > 0.3:
        lines.append("- HMM v3 may be too conservative and may under-warn before higher-vol transitions.")
    if upgrade_success > 0.6:
        lines.append("- HMM v3 appears to detect internal market deterioration before the core HMM.")
    if ties > 0.5:
        lines.append("- HMM v3 changes labels but not enough to materially improve economic risk-bucket outcomes.")
    if not lines:
        lines.append("- Disagreement outcomes are mixed; no dominant pattern yet.")
    return "\n".join(lines)


def build_replay_report_markdown(
    *,
    summary_metrics: pd.DataFrame,
    recent_rows: pd.DataFrame,
    disagreements: pd.DataFrame,
    diagnostics: pd.DataFrame,
    prediction_distribution: pd.DataFrame,
    outcome_distribution: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    economic_summary: pd.DataFrame,
    false_alarms: pd.DataFrame,
    missed_risks: pd.DataFrame,
    disagreement_attribution: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
    geometry_override_cases: pd.DataFrame,
    geometry_false_suppression_cases: pd.DataFrame,
    geometry_false_suppression_analysis: pd.DataFrame,
    geometry_success_cases: pd.DataFrame,
    geometry_smooth_modifier: pd.DataFrame,
) -> str:
    top_disagreements = disagreement_attribution.copy()
    if not top_disagreements.empty:
        if "is_opposite_bucket" in top_disagreements.columns:
            top_disagreements["is_opposite_bucket"] = top_disagreements["is_opposite_bucket"].astype(int)
        else:
            top_disagreements["is_opposite_bucket"] = 0
        top_disagreements["abs_severity_delta"] = top_disagreements["severity_delta"].abs()
        top_disagreements["abs_vix_change_pct_3d"] = top_disagreements["vix_change_pct_3d"].abs()
        top_disagreements["abs_rv21_change_3d"] = top_disagreements["rv21_change_3d"].abs()
        top_disagreements = top_disagreements.sort_values(
            ["is_opposite_bucket", "abs_severity_delta", "abs_vix_change_pct_3d", "abs_rv21_change_3d"],
            ascending=[False, False, False, False],
        ).head(20)
        top_disagreements = top_disagreements[
            [
                "as_of_date",
                "comparison_model",
                "comparison_state",
                "hmm_v3_state",
                "realized_state_1d",
                "realized_state_2d",
                "realized_state_3d",
                "vix_change_pct_3d",
                "avg_pairwise_corr_21d",
                "first_eigenvalue_share_21d",
                "effective_rank_21d",
                "log_det_corr_21d",
                "v3_result",
            ]
        ]
    geometry_override_summary = pd.DataFrame(
        [
            {
                "Type": "v3_downgrade",
                "Count": int((disagreement_attribution["disagreement_type"] == "v3_downgrade").sum()) if not disagreement_attribution.empty else 0,
                "Success Rate": float(disagreement_summary["downgrade_success_rate"].mean()) if not disagreement_summary.empty else 0.0,
                "Notes": "HMM v3 lower-severity than comparison model.",
            },
            {
                "Type": "v3_upgrade",
                "Count": int((disagreement_attribution["disagreement_type"] == "v3_upgrade").sum()) if not disagreement_attribution.empty else 0,
                "Success Rate": float(disagreement_summary["upgrade_success_rate"].mean()) if not disagreement_summary.empty else 0.0,
                "Notes": "HMM v3 higher-severity than comparison model.",
            },
            {
                "Type": "opposite_bucket",
                "Count": int(disagreement_attribution.get("is_opposite_bucket", pd.Series([], dtype=bool)).sum()) if not disagreement_attribution.empty else 0,
                "Success Rate": float(disagreement_summary["v3_bucket_win_rate"].mean()) if not disagreement_summary.empty else 0.0,
                "Notes": "Highest-priority disagreements.",
            },
            {
                "Type": "same_bucket_different_state",
                "Count": int((disagreement_attribution["disagreement_type"] == "v3_same_bucket_different_state").sum()) if not disagreement_attribution.empty else 0,
                "Success Rate": float(disagreement_summary["v3_win_rate"].mean()) if not disagreement_summary.empty else 0.0,
                "Notes": "Label variation inside same economic bucket.",
            },
        ]
    )
    return (
        "# HMM Replay Backtest Report\n\n"
        "## Overall Model Comparison\n\n"
        + _md_table(summary_metrics)
        + "\n\n## Prediction Distribution\n\n"
        + _md_table(prediction_distribution)
        + "\n\n## Outcome Distribution\n\n"
        + _md_table(outcome_distribution)
        + "\n\n## Confusion Matrix by Horizon\n\n"
        + _md_table(confusion_matrix)
        + "\n\n## Economic Score Summary\n\n"
        + _md_table(economic_summary)
        + "\n\n## False Alarms\n\n"
        + _md_table(false_alarms.head(50))
        + "\n\n## Missed Risks\n\n"
        + _md_table(missed_risks.head(50))
        + "\n\n## Recent As-of Date Comparison\n\n"
        + _md_table(recent_rows)
        + "\n\n## Model Disagreement Cases\n\n"
        + _md_table(disagreements)
        + "\n\n## HMM v3 Special Section\n\n"
        "Track whether geometry features improved false vol-expansion avoidance and mid-vol chop detection.\n\n"
        "## Model Usefulness Summary\n\n"
        + _model_usefulness_lines(economic_summary)
        + "\n\n## Disagreement Attribution\n\n"
        "### Disagreement Summary Table\n\n"
        + _md_table(
            disagreement_summary[
                [
                    "comparison_model",
                    "horizon",
                    "total_disagreements",
                    "v3_win_rate",
                    "v3_loss_rate",
                    "tie_rate",
                    "v3_bucket_win_rate",
                ]
            ]
            if not disagreement_summary.empty
            else disagreement_summary
        )
        + "\n\n### Geometry Override Summary\n\n"
        + _md_table(geometry_override_summary)
        + "\n\n### Top 20 Most Important Disagreements\n\n"
        + _md_table(top_disagreements)
        + "\n\n### Plain-English Interpretation\n\n"
        + _disagreement_interpretation(disagreement_summary)
        + "\n\n### Geometry Override Cases\n\n"
        + _md_table(geometry_override_cases.head(50))
        + "\n\n### Geometry False Suppression Cases\n\n"
        + _md_table(geometry_false_suppression_cases.head(50))
        + "\n\n### Geometry False Suppression Analysis\n\n"
        + _md_table(geometry_false_suppression_analysis)
        + "\n\n### Geometry Success Cases\n\n"
        + _md_table(geometry_success_cases.head(50))
        + "\n\n## Geometry Smooth Modifier\n\n"
        + _md_table(geometry_smooth_modifier.head(100))
        + "\n\n## Diagnostics\n\n"
        + _md_table(diagnostics)
        + "\n"
    )


def build_recent_comparison(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> pd.DataFrame:
    out_map = {item["as_of_date"]: item for item in outcomes}
    rows: list[dict[str, Any]] = []
    for prediction in predictions[-30:]:
        outcome = out_map.get(prediction["as_of_date"], {})
        rows.append(
            {
                "As Of": prediction["as_of_date"],
                "Model": prediction["model_name"],
                "Predicted State": prediction["top_state"],
                "T+1 Outcome": outcome.get("realized_regime_label_1d"),
                "T+2 Outcome": outcome.get("realized_regime_label_2d"),
                "T+3 Outcome": outcome.get("realized_regime_label_3d"),
                "Score": prediction.get("score", ""),
            }
        )
    return pd.DataFrame(rows)


def build_disagreement_rows(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(predictions)
    if frame.empty:
        return frame
    pivot = frame.pivot_table(index="as_of_date", columns="model_name", values="top_state", aggfunc="first")
    if "hmm_v3_core_plus_sector_geometry" not in pivot.columns:
        return pd.DataFrame(columns=["as_of_date", "hmm_v3", "other_model", "other_state"])
    rows: list[dict[str, Any]] = []
    for as_of, row in pivot.iterrows():
        v3 = row.get("hmm_v3_core_plus_sector_geometry")
        for column, value in row.items():
            if column == "hmm_v3_core_plus_sector_geometry":
                continue
            if pd.notna(v3) and pd.notna(value) and str(v3) != str(value):
                rows.append(
                    {
                        "as_of_date": as_of,
                        "hmm_v3": v3,
                        "other_model": column,
                        "other_state": value,
                    }
                )
    return pd.DataFrame(rows)
