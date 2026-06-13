from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.hmm_replay.replay_config import ReplayConfig, load_replay_config
from src.backtest.hmm_replay.replay_dataset import filter_date_range, load_feature_store, train_slice
from src.backtest.hmm_replay.replay_outcomes import compute_outcome_record
from src.backtest.hmm_replay.replay_predictions import create_replay_context, generate_prediction_record
from src.backtest.hmm_replay.replay_report import (
    build_disagreement_rows,
    build_recent_comparison,
    build_replay_report_markdown,
)
from src.backtest.hmm_replay.replay_scoring import score_prediction, summarize_scores
from src.backtest.hmm_replay.replay_scoring import (
    build_confusion_matrix_by_horizon,
    build_disagreement_attribution,
    build_disagreement_summary,
    build_false_alarm_and_missed_risk_reports,
    build_geometry_case_files,
    build_geometry_false_suppression_analysis,
    build_outcome_distribution,
    build_prediction_distribution,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _merge_geometry_diagnostics(
    cases_df: pd.DataFrame,
    *,
    prediction_records: list[dict[str, Any]],
) -> pd.DataFrame:
    if cases_df.empty:
        return cases_df
    diagnostics_rows = []
    for record in prediction_records:
        if str(record.get("model_name", "")) != "hmm_v3_1_meta_blend":
            continue
        diagnostics = dict(record.get("model_diagnostics", {}))
        diagnostics_rows.append(
            {
                "as_of_date": str(record.get("as_of_date", "")),
                "geometry_stress_score": diagnostics.get("geometry_stress_score"),
                "downgrade_levels": diagnostics.get("downgrade_levels"),
            }
        )
    if not diagnostics_rows:
        return cases_df
    diagnostics_df = pd.DataFrame(diagnostics_rows).drop_duplicates(subset=["as_of_date"], keep="last")
    return cases_df.merge(diagnostics_df, on="as_of_date", how="left")


def run_hmm_replay(
    *,
    config: ReplayConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    models: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    if as_of_date:
        start = as_of_date
        end = as_of_date
    else:
        start = start_date or config.start_date
        end = end_date or config.end_date
    selected_models = list(models or config.models)
    selected_horizons = list(horizons or config.horizons)

    source = load_feature_store(config.feature_store_path)
    scoped = filter_date_range(source, start_date=start, end_date=end)
    if scoped.empty:
        raise RuntimeError("Replay date range has no rows in the historical feature store.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    artifact_dir = Path(config.artifact_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prediction_records: list[dict[str, Any]] = []
    outcome_records: list[dict[str, Any]] = []
    scored_records: list[dict[str, Any]] = []

    full = source.reset_index(drop=True)
    index_by_date = {str(item): idx for idx, item in enumerate(full["date"])}
    for replay_date in scoped["date"].tolist():
        as_of_text = str(replay_date)
        full_index = index_by_date.get(as_of_text)
        if full_index is None:
            continue
        outcome = compute_outcome_record(full, as_of_index=full_index, horizons=selected_horizons)
        outcome_records.append(outcome)

        training = train_slice(full, as_of_date=as_of_text, lookback_days=config.train_lookback_days)
        for model_name in selected_models:
            context = create_replay_context(as_of_text)
            prediction = generate_prediction_record(
                context=context,
                model_name=model_name,
                train_df=training,
                min_train_rows=config.min_train_rows,
                n_components=config.n_components,
                random_state=config.random_state,
                covariance_type=config.covariance_type,
            )
            prediction_records.append(prediction)
            for horizon in selected_horizons:
                scored_records.append(score_prediction(prediction, outcome, horizon=horizon))

    summary_df = summarize_scores(scored_records)
    prediction_distribution_df = build_prediction_distribution(scored_records)
    outcome_distribution_df = build_outcome_distribution(scored_records)
    confusion_matrix_df = build_confusion_matrix_by_horizon(scored_records)
    false_alarms_df, missed_risks_df = build_false_alarm_and_missed_risk_reports(
        scored_records=scored_records,
        predictions=prediction_records,
        outcomes=outcome_records,
    )
    disagreement_attribution_df = build_disagreement_attribution(
        predictions=prediction_records,
        outcomes=outcome_records,
        scored_records=scored_records,
        horizons=selected_horizons,
    )
    disagreement_summary_df = build_disagreement_summary(
        disagreement_attribution_df,
        horizons=selected_horizons,
    )
    geometry_override_df, geometry_false_suppression_df, geometry_success_df = build_geometry_case_files(
        disagreement_attribution_df,
        horizons=selected_horizons,
    )
    geometry_override_df = _merge_geometry_diagnostics(
        geometry_override_df,
        prediction_records=prediction_records,
    )
    geometry_false_suppression_df = _merge_geometry_diagnostics(
        geometry_false_suppression_df,
        prediction_records=prediction_records,
    )
    geometry_success_df = _merge_geometry_diagnostics(
        geometry_success_df,
        prediction_records=prediction_records,
    )
    geometry_false_suppression_analysis_df = build_geometry_false_suppression_analysis(
        geometry_false_suppression_df,
        horizons=selected_horizons,
    )
    economic_summary_df = summary_df.copy()
    geometry_modifier_df = pd.DataFrame(
        [
            {
                "as_of_date": row.get("as_of_date"),
                "core_vol_risk_score": dict(row.get("model_diagnostics", {})).get("core_vol_risk_score"),
                "geometry_stress_score": dict(row.get("model_diagnostics", {})).get("geometry_stress_score"),
                "final_risk_score": dict(row.get("model_diagnostics", {})).get("final_risk_score"),
                "core_vol_state": dict(row.get("model_diagnostics", {})).get("core_vol_state"),
                "final_regime": dict(row.get("model_diagnostics", {})).get("final_regime"),
                "downgrade_levels": dict(row.get("model_diagnostics", {})).get("downgrade_levels"),
                "downgrade_cap_applied": dict(row.get("model_diagnostics", {})).get("downgrade_cap_applied"),
            }
            for row in prediction_records
            if str(row.get("model_name", "")) == "hmm_v3_1_meta_blend"
        ]
    )
    recent_df = build_recent_comparison(prediction_records, outcome_records)
    disagreement_df = build_disagreement_rows(prediction_records)
    diagnostics_df = pd.DataFrame(
        [
            {
                "as_of_date": row["as_of_date"],
                "model_name": row["model_name"],
                "converged": dict(row.get("model_diagnostics", {})).get("converged"),
                "training_row_count": dict(row.get("model_diagnostics", {})).get("training_row_count"),
                "training_end_date": dict(row.get("model_diagnostics", {})).get("training_end_date"),
                "warnings": " | ".join(list(row.get("warnings", []))),
            }
            for row in prediction_records
        ]
    )

    prediction_path = artifact_dir / "prediction_records.jsonl"
    outcome_path = artifact_dir / "outcome_records.jsonl"
    scored_path = artifact_dir / "scored_records.jsonl"
    summary_path = output_dir / "summary_metrics.csv"
    prediction_distribution_path = output_dir / "prediction_distribution.csv"
    outcome_distribution_path = output_dir / "outcome_distribution.csv"
    confusion_matrix_path = output_dir / "confusion_matrix_by_horizon.csv"
    economic_summary_path = output_dir / "economic_score_summary.csv"
    false_alarms_path = output_dir / "false_alarms.csv"
    missed_risks_path = output_dir / "missed_risks.csv"
    disagreement_attribution_path = output_dir / "disagreement_attribution.csv"
    disagreement_summary_path = output_dir / "disagreement_summary.csv"
    geometry_override_path = output_dir / "geometry_override_cases.csv"
    geometry_false_suppression_path = output_dir / "geometry_false_suppression_cases.csv"
    geometry_false_suppression_analysis_path = output_dir / "geometry_false_suppression_analysis.csv"
    geometry_success_path = output_dir / "geometry_success_cases.csv"
    geometry_smooth_modifier_path = output_dir / "geometry_smooth_modifier.csv"
    report_path = output_dir / "replay_report.md"

    _write_jsonl(prediction_path, prediction_records)
    _write_jsonl(outcome_path, outcome_records)
    _write_jsonl(scored_path, scored_records)
    summary_df.to_csv(summary_path, index=False)
    prediction_distribution_df.to_csv(prediction_distribution_path, index=False)
    outcome_distribution_df.to_csv(outcome_distribution_path, index=False)
    confusion_matrix_df.to_csv(confusion_matrix_path, index=False)
    economic_summary_df.to_csv(economic_summary_path, index=False)
    false_alarms_df.to_csv(false_alarms_path, index=False)
    missed_risks_df.to_csv(missed_risks_path, index=False)
    disagreement_attribution_df.to_csv(disagreement_attribution_path, index=False)
    disagreement_summary_df.to_csv(disagreement_summary_path, index=False)
    geometry_override_df.to_csv(geometry_override_path, index=False)
    geometry_false_suppression_df.to_csv(geometry_false_suppression_path, index=False)
    geometry_false_suppression_analysis_df.to_csv(geometry_false_suppression_analysis_path, index=False)
    geometry_success_df.to_csv(geometry_success_path, index=False)
    geometry_modifier_df.to_csv(geometry_smooth_modifier_path, index=False)
    report_path.write_text(
        build_replay_report_markdown(
            summary_metrics=summary_df,
            recent_rows=recent_df,
            disagreements=disagreement_df,
            diagnostics=diagnostics_df,
            prediction_distribution=prediction_distribution_df,
            outcome_distribution=outcome_distribution_df,
            confusion_matrix=confusion_matrix_df,
            economic_summary=economic_summary_df,
            false_alarms=false_alarms_df,
            missed_risks=missed_risks_df,
            disagreement_attribution=disagreement_attribution_df,
            disagreement_summary=disagreement_summary_df,
            geometry_override_cases=geometry_override_df,
            geometry_false_suppression_cases=geometry_false_suppression_df,
            geometry_false_suppression_analysis=geometry_false_suppression_analysis_df,
            geometry_success_cases=geometry_success_df,
            geometry_smooth_modifier=geometry_modifier_df,
        ),
        encoding="utf-8",
    )
    return {
        "prediction_records_path": str(prediction_path),
        "outcome_records_path": str(outcome_path),
        "scored_records_path": str(scored_path),
        "summary_metrics_path": str(summary_path),
        "prediction_distribution_path": str(prediction_distribution_path),
        "outcome_distribution_path": str(outcome_distribution_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "economic_summary_path": str(economic_summary_path),
        "false_alarms_path": str(false_alarms_path),
        "missed_risks_path": str(missed_risks_path),
        "disagreement_attribution_path": str(disagreement_attribution_path),
        "disagreement_summary_path": str(disagreement_summary_path),
        "geometry_override_path": str(geometry_override_path),
        "geometry_false_suppression_path": str(geometry_false_suppression_path),
        "geometry_false_suppression_analysis_path": str(geometry_false_suppression_analysis_path),
        "geometry_success_path": str(geometry_success_path),
        "geometry_smooth_modifier_path": str(geometry_smooth_modifier_path),
        "report_path": str(report_path),
        "summary_metrics": summary_df.to_dict(orient="records"),
        "geometry_false_suppression_analysis": geometry_false_suppression_analysis_df.to_dict(orient="records"),
    }


def _parse_list_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list_arg(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic HMM replay backtester.")
    parser.add_argument("--config", required=True, help="Path to replay config YAML.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--horizons", default=None, help="Comma-separated integer horizons.")
    args = parser.parse_args()

    config = load_replay_config(args.config)
    result = run_hmm_replay(
        config=config,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        models=_parse_list_arg(args.models) or None,
        horizons=_parse_int_list_arg(args.horizons) or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
