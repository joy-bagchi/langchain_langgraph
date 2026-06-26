from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.hmm_replay.replay_config import ReplayConfig, load_replay_config
from src.backtest.hmm_replay.replay_dataset import filter_date_range, load_feature_store
from src.backtest.hmm_replay.replay_outcomes import compute_outcome_record
from src.backtest.hmm_replay.preflight import run_replay_preflight, write_preflight_failure_artifacts
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

TUNING_DEFAULT_MODELS = ("hmm_v4_path_aware_meta", "hmm_v3_1_meta_blend")
TUNING_DEFAULT_HORIZONS = (1, 3)
TUNING_DEFAULT_LOOKBACK_DAYS = 756
TUNING_DEFAULT_MIN_TRAIN_ROWS = 504
TUNING_DEFAULT_MAX_REPLAY_DATES = 75
TUNING_DEFAULT_WINDOW_BDAYS = 90
BASE_CACHEABLE_MODELS = {
    "heuristic",
    "hmm_v1_core",
    "hmm_v2_core_plus_sector_corr",
    "hmm_v3_core_plus_sector_geometry",
    "hmm_v3_1_meta_blend",
}


def _json_safe_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _build_run_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    log_path = output_dir / "backtest_run.log"
    logger_name = f"hmm_replay_{output_dir.name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, log_path


def _append_jsonl_line(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=_json_safe_default))
    handle.write("\n")


def _shutdown_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_safe_default))
            handle.write("\n")


def _build_prediction_cache_key(
    *,
    feature_store_path: Path,
    model_name: str,
    run_mode: str,
    train_lookback_days: int,
    min_train_rows: int,
    n_components: int,
    covariance_type: str,
    random_state: int,
) -> str:
    stat = feature_store_path.stat()
    payload = {
        "feature_store_path": str(feature_store_path.resolve()),
        "feature_store_size": int(stat.st_size),
        "feature_store_mtime_ns": int(stat.st_mtime_ns),
        "model_name": model_name,
        "run_mode": run_mode,
        "train_lookback_days": int(train_lookback_days),
        "min_train_rows": int(min_train_rows),
        "n_components": int(n_components),
        "covariance_type": str(covariance_type),
        "random_state": int(random_state),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_prediction_cache(cache_file: Path) -> dict[str, dict[str, Any]]:
    if not cache_file.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with cache_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            as_of = str(row.get("as_of_date", "")).strip()
            if not as_of:
                continue
            entries[as_of] = row
    return entries


def _compact_prediction_record(prediction: dict[str, Any]) -> dict[str, Any]:
    feature_snapshot = dict(prediction.get("feature_snapshot", {}))
    path_features = dict(prediction.get("path_features", {}))
    diagnostics = dict(prediction.get("model_diagnostics", {}))
    compact_feature_keys = [
        "vix",
        "vvix",
        "vvix_vix_ratio",
        "vvix_vix_z_22d",
        "vix_vix3m_ratio",
        "vix9d_vix_ratio",
        "term_structure_slope",
        "realized_vol_5d",
        "realized_vol_21d",
        "spy_return_1d",
        "drawdown_21d",
        "trend_persistence_21d",
        "avg_pairwise_corr_21d",
        "first_eigenvalue_share_21d",
        "effective_rank_21d",
        "log_det_corr_21d",
    ]
    compact_diagnostic_keys = [
        "converged",
        "training_row_count",
        "training_end_date",
        "fallback_used",
        "target_horizon",
        "predicted_risk_bucket",
        "path_aware_estimator",
        "feature_families_used",
        "core_vol_risk_score",
        "geometry_stress_score",
        "final_risk_score",
        "core_vol_state",
        "final_regime",
        "downgrade_levels",
        "downgrade_cap_applied",
        "grouped_feature_family_importance",
        "single_class_training",
        "positive_label_rate",
    ]
    return {
        "run_id": prediction.get("run_id"),
        "as_of_date": prediction.get("as_of_date"),
        "model_name": prediction.get("model_name"),
        "top_state": prediction.get("top_state"),
        "state_probabilities": dict(prediction.get("state_probabilities", {})),
        "transition_probabilities": dict(prediction.get("transition_probabilities", {})),
        "policy_output": dict(prediction.get("policy_output", {})),
        "feature_snapshot": {
            key: feature_snapshot.get(key)
            for key in compact_feature_keys
            if key in feature_snapshot
        },
        "path_features": dict(path_features),
        "model_diagnostics": {
            key: diagnostics.get(key)
            for key in compact_diagnostic_keys
            if key in diagnostics
        },
        "warnings": list(prediction.get("warnings", []))[:6],
        "top_feature_importances": list(prediction.get("top_feature_importances", []))[:10],
    }


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


def _build_path_aware_summary(
    *,
    prediction_records: list[dict[str, Any]],
    scored_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = [row for row in prediction_records if str(row.get("model_name", "")) == "hmm_v4_path_aware_meta"]
    if not prediction_rows:
        return pd.DataFrame(), pd.DataFrame()
    diag_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        diagnostics = dict(row.get("model_diagnostics", {}))
        path_features = dict(diagnostics.get("path_features", row.get("path_features", {})) or {})
        top_importances = list(diagnostics.get("top_feature_importances", row.get("top_feature_importances", [])) or [])
        diag_rows.append(
            {
                "as_of_date": row.get("as_of_date"),
                "target_horizon": diagnostics.get("target_horizon"),
                "training_row_count": diagnostics.get("training_row_count"),
                "path_aware_estimator": diagnostics.get("path_aware_estimator"),
                "fallback_used": diagnostics.get("fallback_used"),
                "feature_families_used": " | ".join(list(diagnostics.get("feature_families_used", []))),
                "predicted_risk_bucket": diagnostics.get("predicted_risk_bucket"),
                "geometry_stress_score": path_features.get("geometry_stress_score"),
                "geometry_stress_delta_5d": path_features.get("geometry_stress_score_delta_5d"),
                "geometry_stress_curvature_5_10": path_features.get("geometry_stress_score_curvature_5_10"),
                "vol_geometry_gap": path_features.get("vol_geometry_gap"),
                "vol_geometry_diverging": path_features.get("vol_geometry_diverging"),
                "top_feature_importances": " | ".join(
                    f"{item.get('feature')}={float(item.get('importance', 0.0)):.3f}" for item in top_importances[:5]
                ),
            }
        )
    path_diag_df = pd.DataFrame(diag_rows)

    scored = pd.DataFrame([row for row in scored_records if str(row.get("model_name", "")) == "hmm_v4_path_aware_meta"])
    if scored.empty:
        return path_diag_df, pd.DataFrame()
    if "risk_bucket_correct" not in scored.columns:
        return path_diag_df, pd.DataFrame()
    merged = scored.merge(path_diag_df, on="as_of_date", how="left")
    wins = merged[merged["risk_bucket_correct"] == True]  # noqa: E712
    losses = merged[merged["risk_bucket_correct"] == False]  # noqa: E712

    def _avg(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").mean())

    rows = []
    for label, frame in (("wins", wins), ("losses", losses)):
        rows.append(
            {
                "segment": label,
                "avg_geometry_stress_delta_5d": _avg(frame, "geometry_stress_delta_5d"),
                "avg_geometry_curvature_5_10": _avg(frame, "geometry_stress_curvature_5_10"),
                "avg_vol_geometry_gap": _avg(frame, "vol_geometry_gap"),
                "success_rate_when_geometry_accelerating": float(
                    ((frame["geometry_stress_curvature_5_10"] > 0) & (frame["risk_bucket_correct"] == True)).mean()  # noqa: E712
                )
                if not frame.empty and "geometry_stress_curvature_5_10" in frame.columns
                else 0.0,
                "success_rate_when_geometry_decelerating": float(
                    ((frame["geometry_stress_curvature_5_10"] <= 0) & (frame["risk_bucket_correct"] == True)).mean()  # noqa: E712
                )
                if not frame.empty and "geometry_stress_curvature_5_10" in frame.columns
                else 0.0,
                "success_rate_when_vol_geometry_diverge": float(
                    ((frame["vol_geometry_diverging"] == 1) & (frame["risk_bucket_correct"] == True)).mean()  # noqa: E712
                )
                if not frame.empty and "vol_geometry_diverging" in frame.columns
                else 0.0,
                "success_rate_when_vol_geometry_confirm": float(
                    ((pd.to_numeric(frame["vol_geometry_gap"], errors="coerce").abs() < 0.15) & (frame["risk_bucket_correct"] == True)).mean()  # noqa: E712
                )
                if not frame.empty and "vol_geometry_gap" in frame.columns
                else 0.0,
            }
        )
    return path_diag_df, pd.DataFrame(rows)


def _build_top_feature_importances(prediction_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in prediction_records:
        model_name = str(record.get("model_name", "")).strip()
        for item in list(record.get("top_feature_importances", [])):
            feature = str(dict(item).get("feature", "")).strip()
            importance = pd.to_numeric(dict(item).get("importance"), errors="coerce")
            if not model_name or not feature or pd.isna(importance):
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model_name", "feature", "avg_importance", "median_importance", "count"])

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["model_name", "feature"], as_index=False)
        .agg(
            avg_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            max_importance=("importance", "max"),
            count=("importance", "size"),
        )
        .sort_values(["model_name", "avg_importance"], ascending=[True, False])
    )
    return grouped.groupby("model_name", group_keys=False).head(25).reset_index(drop=True)


def _build_compact_summary_markdown(
    *,
    run_mode: str,
    selected_models: list[str],
    selected_horizons: list[int],
    eligible_dates_count: int,
    total_prediction_records: int,
    total_scored_records: int,
    cache_hits: int,
    cache_misses: int,
    summary_df: pd.DataFrame,
    runtime_profile: dict[str, Any],
    peak_rss_mb: float | None,
) -> str:
    lines = [
        "# HMM Replay Compact Summary",
        "",
        f"- Run mode: `{run_mode}`",
        f"- Models: `{', '.join(selected_models)}`",
        f"- Horizons: `{', '.join(str(item) for item in selected_horizons)}`",
        f"- Replay dates evaluated: `{eligible_dates_count}`",
        f"- Prediction records: `{total_prediction_records}`",
        f"- Scored records: `{total_scored_records}`",
        f"- Cache hits: `{cache_hits}`",
        f"- Cache misses: `{cache_misses}`",
        f"- Peak RSS memory (MB): `{peak_rss_mb if peak_rss_mb is not None else 'n/a'}`",
        "",
        "## Runtime Profile (seconds)",
    ]
    for key in [
        "feature_load_seconds",
        "feature_engineering_seconds",
        "model_fit_seconds",
        "prediction_seconds",
        "scoring_seconds",
        "report_generation_seconds",
        "total_seconds",
    ]:
        lines.append(f"- {key}: `{runtime_profile.get(key, 0.0)}`")
    lines.append("")
    lines.append("## Summary Metrics Preview")
    if summary_df.empty:
        lines.append("")
        lines.append("_No summary rows were produced._")
    else:
        lines.append("")
        lines.append(summary_df.head(25).to_markdown(index=False))
    lines.append("")
    lines.append(
        "_Tuning mode is intended for fast behavior discovery only. Use testing mode for final validation evidence._"
    )
    lines.append("")
    return "\n".join(lines)


def run_hmm_replay(
    *,
    config: ReplayConfig,
    run_mode: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    models: list[str] | None = None,
    horizons: list[int] | None = None,
    lightweight_mode: bool = False,
) -> dict[str, Any]:
    effective_run_mode = str(run_mode or config.run_mode or "testing").strip().lower()
    if effective_run_mode not in {"tuning", "testing"}:
        raise RuntimeError(f"Unsupported run_mode: {effective_run_mode}. Expected 'tuning' or 'testing'.")
    effective_lightweight_mode = bool(lightweight_mode) or effective_run_mode == "tuning"
    stage_profile_seconds: dict[str, float] = {
        "feature_load_seconds": 0.0,
        "feature_engineering_seconds": 0.0,
        "model_fit_seconds": 0.0,
        "prediction_seconds": 0.0,
        "scoring_seconds": 0.0,
        "report_generation_seconds": 0.0,
    }

    source = pd.DataFrame()
    if as_of_date:
        start = as_of_date
        end = as_of_date
    else:
        start = start_date or config.start_date
        end = end_date or config.end_date
    if effective_run_mode == "tuning" and not as_of_date:
        end = str(pd.Timestamp.utcnow().date()) if str(end).strip().lower() == "latest" else str(end)
    elif str(end).strip().lower() == "latest":
        end = str(pd.Timestamp.utcnow().date())
    selected_models = list(models or config.models)
    selected_horizons = list(horizons or config.horizons)
    effective_train_lookback_days = int(config.train_lookback_days)
    effective_min_train_rows = int(config.min_train_rows)
    max_replay_dates: int | None = None
    if effective_run_mode == "tuning":
        if not models:
            available_models = list(config.models or [])
            defaults: list[str] = [name for name in TUNING_DEFAULT_MODELS if name in available_models]
            if not defaults and available_models:
                defaults = available_models[:2]
            selected_models = defaults or list(TUNING_DEFAULT_MODELS)
        if not horizons:
            selected_horizons = list(TUNING_DEFAULT_HORIZONS)
        effective_train_lookback_days = int(TUNING_DEFAULT_LOOKBACK_DAYS)
        effective_min_train_rows = int(TUNING_DEFAULT_MIN_TRAIN_ROWS)
        max_replay_dates = int(TUNING_DEFAULT_MAX_REPLAY_DATES)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config.output_dir) / f"run_{timestamp}"
    artifact_dir = Path(config.artifact_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger, run_log_path = _build_run_logger(output_dir)
    started_at = time.perf_counter()
    logger.info(
        "Replay start | config=%s | run_mode=%s | start=%s | end=%s | as_of=%s | models=%s | horizons=%s | lightweight_mode=%s",
        str(config.feature_store_path),
        effective_run_mode,
        str(start),
        str(end),
        str(as_of_date),
        ",".join(selected_models),
        ",".join(str(item) for item in selected_horizons),
        str(bool(effective_lightweight_mode)).lower(),
    )

    last_context = {"as_of_date": "", "model": ""}
    prediction_cache_append_handles: dict[str, Any] = {}
    try:
        feature_load_started = time.perf_counter()
        source = load_feature_store(config.feature_store_path)
        stage_profile_seconds["feature_load_seconds"] += time.perf_counter() - feature_load_started
        if effective_run_mode == "tuning" and not as_of_date:
            latest_source_date = pd.to_datetime(source["date"].max()).date()
            tuning_start = pd.bdate_range(end=latest_source_date, periods=TUNING_DEFAULT_WINDOW_BDAYS).min().date()
            start = str(max(pd.to_datetime(start).date(), tuning_start))
            end = str(latest_source_date)

        resolved_end = str(source["date"].max()) if str(end).strip().lower() == "latest" else str(end)
        preflight = run_replay_preflight(
            frame=source,
            config=config,
            requested_start_date=str(start),
            requested_end_date=resolved_end,
        )
        if not preflight.ok:
            report_path, missing_csv_path = write_preflight_failure_artifacts(
                output_dir=output_dir,
                preflight=preflight,
            )
            raise RuntimeError(
                "Replay preflight failed. "
                f"Report: {report_path}. Missing-data CSV: {missing_csv_path}. "
                + " | ".join(preflight.messages)
            )

        scoped = filter_date_range(source, start_date=preflight.requested_start_date, end_date=preflight.actual_end_date)
        if scoped.empty:
            raise RuntimeError("Replay date range has no rows in the historical feature store.")

        prediction_records: list[dict[str, Any]] = []
        outcome_records: list[dict[str, Any]] = []
        scored_records: list[dict[str, Any]] = []

        full = source.reset_index(drop=True)
        index_by_date = {str(item): idx for idx, item in enumerate(full["date"])}
        minimum_train_rows = int(effective_min_train_rows)
        earliest_eligible_index = max(minimum_train_rows - 1, 0)
        eligible_replay_dates = [
            replay_date
            for replay_date in scoped["date"].tolist()
            if int(index_by_date.get(str(replay_date), -1)) >= earliest_eligible_index
        ]
        if max_replay_dates is not None and len(eligible_replay_dates) > int(max_replay_dates):
            eligible_replay_dates = eligible_replay_dates[-int(max_replay_dates) :]
        if not eligible_replay_dates:
            earliest_available = str(full["date"].min()) if not full.empty else ""
            latest_available = str(full["date"].max()) if not full.empty else ""
            total_rows = int(len(full))
            raise RuntimeError(
                "Replay date range does not contain any dates with sufficient training history. "
                f"required_min_train_rows={minimum_train_rows}, available_rows={total_rows}, "
                f"earliest_available_date={earliest_available}, latest_available_date={latest_available}."
            )
        logger.info(
            "Eligible replay dates: %d | first=%s | last=%s | min_train_rows=%d | train_lookback_days=%d",
            len(eligible_replay_dates),
            str(eligible_replay_dates[0]),
            str(eligible_replay_dates[-1]),
            minimum_train_rows,
            int(effective_train_lookback_days),
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
        path_aware_diagnostics_path = output_dir / "path_aware_meta_learner.csv"
        path_feature_diagnostics_path = output_dir / "path_feature_diagnostics.csv"
        top_feature_importances_path = output_dir / "top_feature_importances.csv"
        compact_summary_path = output_dir / "compact_summary.md"
        report_path = output_dir / "replay_report.md"

        precomputed_path_aware_cache = None
        if "hmm_v4_path_aware_meta" in selected_models:
            feature_engineering_started = time.perf_counter()
            from src.agents.hmm_v4_path_aware_meta_agent import load_hmm_v4_config
            from src.backtest.hmm_replay.path_aware_dataset import build_path_aware_precomputed_cache

            v4_config = load_hmm_v4_config()
            precomputed_path_aware_cache = build_path_aware_precomputed_cache(
                full,
                feature_windows=list(v4_config.feature_windows),
                geometry_stress_lookback=int(v4_config.geometry_stress_lookback),
            )
            logger.info(
                "Precomputed path-aware cache | rows=%d | features_families=%d",
                int(len(precomputed_path_aware_cache.enriched_frame)),
                int(len(precomputed_path_aware_cache.feature_families)),
            )
            stage_profile_seconds["feature_engineering_seconds"] += time.perf_counter() - feature_engineering_started

        cache_root = (Path(config.artifact_dir) / "_prediction_cache").resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_hits = 0
        cache_misses = 0
        prediction_cache_maps: dict[str, dict[str, dict[str, Any]]] = {}
        feature_store_path_resolved = Path(config.feature_store_path).resolve()
        for model_name in selected_models:
            if model_name not in BASE_CACHEABLE_MODELS:
                continue
            cache_key = _build_prediction_cache_key(
                feature_store_path=feature_store_path_resolved,
                model_name=model_name,
                run_mode=effective_run_mode,
                train_lookback_days=effective_train_lookback_days,
                min_train_rows=effective_min_train_rows,
                n_components=config.n_components,
                covariance_type=config.covariance_type,
                random_state=config.random_state,
            )
            cache_file = cache_root / f"{model_name}_{cache_key}.jsonl"
            prediction_cache_maps[model_name] = _load_prediction_cache(cache_file)
            prediction_cache_append_handles[model_name] = cache_file.open("a", encoding="utf-8")

        peak_rss_mb = _process_rss_mb() or 0.0
        slowest_dates: list[tuple[float, str]] = []
        outcome_by_date: dict[str, dict[str, Any]] = {}
        with prediction_path.open("w", encoding="utf-8") as prediction_handle, outcome_path.open(
            "w", encoding="utf-8"
        ) as outcome_handle, scored_path.open("w", encoding="utf-8") as scored_handle:
            for replay_date in eligible_replay_dates:
                as_of_text = str(replay_date)
                full_index = index_by_date.get(as_of_text)
                if full_index is None:
                    continue
                outcome = compute_outcome_record(full, as_of_index=full_index, horizons=selected_horizons)
                outcome_by_date[as_of_text] = outcome
                outcome_records.append(outcome)
                _append_jsonl_line(outcome_handle, outcome)

            for date_index, replay_date in enumerate(eligible_replay_dates, start=1):
                date_started = time.perf_counter()
                as_of_text = str(replay_date)
                last_context["as_of_date"] = as_of_text
                full_index = index_by_date.get(as_of_text)
                if full_index is None:
                    continue
                outcome = outcome_by_date.get(as_of_text)
                if outcome is None:
                    continue

                train_start_index = max(0, int(full_index) - int(effective_train_lookback_days) + 1)
                training = full.iloc[train_start_index : int(full_index) + 1]
                for model_name in selected_models:
                    last_context["model"] = model_name
                    compact_prediction = None
                    cached_for_model = prediction_cache_maps.get(model_name)
                    if cached_for_model is not None and as_of_text in cached_for_model:
                        compact_prediction = dict(cached_for_model[as_of_text])
                        cache_hits += 1
                    else:
                        prediction_started = time.perf_counter()
                        context = create_replay_context(as_of_text)
                        prediction = generate_prediction_record(
                            context=context,
                            model_name=model_name,
                            train_df=training,
                            min_train_rows=effective_min_train_rows,
                            n_components=config.n_components,
                            random_state=config.random_state,
                            covariance_type=config.covariance_type,
                            precomputed_path_aware_cache=precomputed_path_aware_cache,
                        )
                        stage_profile_seconds["prediction_seconds"] += time.perf_counter() - prediction_started
                        if model_name != "heuristic":
                            stage_profile_seconds["model_fit_seconds"] += time.perf_counter() - prediction_started
                        compact_prediction = _compact_prediction_record(prediction)
                        if cached_for_model is not None:
                            cached_for_model[as_of_text] = dict(compact_prediction)
                            _append_jsonl_line(prediction_cache_append_handles[model_name], compact_prediction)
                        cache_misses += 1
                        del prediction
                        del context
                    prediction_records.append(compact_prediction)
                    _append_jsonl_line(prediction_handle, compact_prediction)
                    for horizon in selected_horizons:
                        scoring_started = time.perf_counter()
                        scored_row = score_prediction(compact_prediction, outcome, horizon=horizon)
                        stage_profile_seconds["scoring_seconds"] += time.perf_counter() - scoring_started
                        scored_records.append(scored_row)
                        _append_jsonl_line(scored_handle, scored_row)
                    del compact_prediction
                del training
                elapsed_for_date = time.perf_counter() - date_started
                slowest_dates.append((float(elapsed_for_date), as_of_text))
                if len(slowest_dates) > 15:
                    slowest_dates = sorted(slowest_dates, key=lambda item: item[0], reverse=True)[:10]

                if date_index % 25 == 0 or date_index == len(eligible_replay_dates):
                    gc.collect()
                    rss_mb = _process_rss_mb()
                    if rss_mb is not None:
                        peak_rss_mb = max(float(peak_rss_mb), float(rss_mb))
                    elapsed = time.perf_counter() - started_at
                    obj_count = len(gc.get_objects())
                    logger.info(
                        "Progress %d/%d dates | as_of=%s | predictions=%d | scores=%d | elapsed=%.1fs | rss_mb=%s | obj_count=%d | cache_hits=%d | cache_misses=%d",
                        date_index,
                        len(eligible_replay_dates),
                        as_of_text,
                        len(prediction_records),
                        len(scored_records),
                        elapsed,
                        (f"{rss_mb:.1f}" if rss_mb is not None else "n/a"),
                        obj_count,
                        int(cache_hits),
                        int(cache_misses),
                    )
                    (artifact_dir / "progress.json").write_text(
                        json.dumps(
                            {
                                "completed_dates": date_index,
                                "total_dates": len(eligible_replay_dates),
                                "last_as_of_date": as_of_text,
                                "prediction_records": len(prediction_records),
                                "scored_records": len(scored_records),
                                "elapsed_seconds": round(float(elapsed), 3),
                                "rss_mb": round(float(rss_mb), 3) if rss_mb is not None else None,
                                "peak_rss_mb": round(float(peak_rss_mb), 3) if peak_rss_mb else None,
                                "object_count": int(obj_count),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            del outcome_by_date
            del precomputed_path_aware_cache
        if not scored_records:
            raise RuntimeError(
                "Replay produced zero scored rows after warmup/date filtering. "
                "Adjust start/end dates, selected models, or horizons."
            )

        summary_df = summarize_scores(scored_records)
        prediction_distribution_df = build_prediction_distribution(scored_records)
        outcome_distribution_df = build_outcome_distribution(scored_records)
        confusion_matrix_df = build_confusion_matrix_by_horizon(scored_records)
        report_started = time.perf_counter()
        if effective_lightweight_mode:
            false_alarms_df = pd.DataFrame()
            missed_risks_df = pd.DataFrame()
            disagreement_attribution_df = pd.DataFrame()
            disagreement_summary_df = pd.DataFrame()
            geometry_override_df = pd.DataFrame()
            geometry_false_suppression_df = pd.DataFrame()
            geometry_success_df = pd.DataFrame()
            geometry_false_suppression_analysis_df = pd.DataFrame(
                [{"segment": "overall", "metric": "case_count", "category": "", "value": 0.0}]
            )
            path_aware_diagnostics_df, path_aware_feature_diagnostics_df = _build_path_aware_summary(
                prediction_records=prediction_records,
                scored_records=scored_records,
            )
            economic_summary_df = summary_df.copy()
            geometry_modifier_df = pd.DataFrame()
            recent_df = pd.DataFrame()
            disagreement_df = pd.DataFrame()
            diagnostics_df = pd.DataFrame()
        else:
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
            path_aware_diagnostics_df, path_aware_feature_diagnostics_df = _build_path_aware_summary(
                prediction_records=prediction_records,
                scored_records=scored_records,
            )
            if config.require_10y_replay and "hmm_v4_path_aware_meta" in selected_models:
                fallback_flags = pd.to_numeric(
                    path_aware_diagnostics_df.get("fallback_used", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0.0)
                fallback_rate = float(fallback_flags.mean()) if len(fallback_flags) else 0.0
                if fallback_rate > preflight.fallback_rate_limit:
                    raise RuntimeError(
                        "10-year HMMv4 replay failed because fallback usage stayed above the allowed limit. "
                        f"fallback_rate={fallback_rate:.4f}, limit={preflight.fallback_rate_limit:.4f}."
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

        top_feature_importances_df = _build_top_feature_importances(prediction_records)
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
        path_aware_diagnostics_df.to_csv(path_aware_diagnostics_path, index=False)
        path_aware_feature_diagnostics_df.to_csv(path_feature_diagnostics_path, index=False)
        top_feature_importances_df.to_csv(top_feature_importances_path, index=False)
        if effective_lightweight_mode:
            report_path.write_text(
                (
                    "# HMM Replay Backtest Report (Lightweight)\n\n"
                    "Detailed disagreement and geometry sections were skipped in lightweight mode.\n\n"
                    f"- Models: {', '.join(selected_models)}\n"
                    f"- Horizons: {', '.join(str(item) for item in selected_horizons)}\n"
                    f"- Rows scored: {len(scored_records)}\n\n"
                    "## Summary Metrics\n\n"
                    + summary_df.to_markdown(index=False)
                    + "\n"
                ),
                encoding="utf-8",
            )
        else:
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
                    path_aware_meta_learner=path_aware_diagnostics_df,
                    path_feature_diagnostics=path_aware_feature_diagnostics_df,
                ),
                encoding="utf-8",
            )
        stage_profile_seconds["report_generation_seconds"] += time.perf_counter() - report_started
        elapsed_total = time.perf_counter() - started_at
        slowest_dates = sorted(slowest_dates, key=lambda item: item[0], reverse=True)[:10]
        logger.info(
            "Replay resource summary | peak_rss_mb=%s | slowest_dates=%s",
            (f"{peak_rss_mb:.1f}" if peak_rss_mb else "n/a"),
            ", ".join(f"{day}:{seconds:.2f}s" for seconds, day in slowest_dates),
        )
        logger.info(
            "Replay completed | dates=%d | predictions=%d | scores=%d | elapsed=%.1fs",
            len(eligible_replay_dates),
            len(prediction_records),
            len(scored_records),
            elapsed_total,
        )
        runtime_profile_payload = {
            **{key: round(float(value), 6) for key, value in stage_profile_seconds.items()},
            "total_seconds": round(float(elapsed_total), 6),
            "peak_rss_mb": round(float(peak_rss_mb), 3) if peak_rss_mb else None,
        }
        compact_summary_path.write_text(
            _build_compact_summary_markdown(
                run_mode=effective_run_mode,
                selected_models=selected_models,
                selected_horizons=selected_horizons,
                eligible_dates_count=len(eligible_replay_dates),
                total_prediction_records=len(prediction_records),
                total_scored_records=len(scored_records),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                summary_df=summary_df,
                runtime_profile=runtime_profile_payload,
                peak_rss_mb=(round(float(peak_rss_mb), 3) if peak_rss_mb else None),
            ),
            encoding="utf-8",
        )
        preview_row_limit = 50 if effective_run_mode == "tuning" else 100
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
            "path_aware_meta_learner_path": str(path_aware_diagnostics_path),
            "path_feature_diagnostics_path": str(path_feature_diagnostics_path),
            "top_feature_importances_path": str(top_feature_importances_path),
            "compact_summary_path": str(compact_summary_path),
            "report_path": str(report_path),
            "run_log_path": str(run_log_path),
            "peak_rss_mb": round(float(peak_rss_mb), 3) if peak_rss_mb else None,
            "run_mode": effective_run_mode,
            "slowest_dates": [
                {"as_of_date": day, "elapsed_seconds": round(float(seconds), 6)}
                for seconds, day in slowest_dates
            ],
            "total_prediction_records": int(len(prediction_records)),
            "total_scored_records": int(len(scored_records)),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "lightweight_mode": bool(effective_lightweight_mode),
            "runtime_profile": runtime_profile_payload,
            "summary_metrics": summary_df.head(preview_row_limit).to_dict(orient="records"),
            "prediction_distribution": prediction_distribution_df.head(preview_row_limit).to_dict(orient="records"),
            "geometry_false_suppression_analysis": (
                geometry_false_suppression_analysis_df.head(preview_row_limit).to_dict(orient="records")
            ),
            "path_aware_meta_learner": path_aware_diagnostics_df.head(preview_row_limit).to_dict(orient="records"),
            "path_feature_diagnostics": (
                path_aware_feature_diagnostics_df.head(preview_row_limit).to_dict(orient="records")
            ),
            "top_feature_importances": top_feature_importances_df.head(preview_row_limit).to_dict(orient="records"),
        }
    except Exception as exc:
        elapsed_total = time.perf_counter() - started_at
        failure_path = output_dir / "failure_traceback.txt"
        failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception(
            "Replay failed | as_of=%s | model=%s | elapsed=%.1fs",
            last_context.get("as_of_date", ""),
            last_context.get("model", ""),
            elapsed_total,
        )
        raise RuntimeError(
            "Backtest failed. "
            f"run_log={run_log_path} | traceback={failure_path} | "
            f"last_as_of={last_context.get('as_of_date', '')} | "
            f"last_model={last_context.get('model', '')}. "
            f"Original error: {exc}"
        ) from exc
    finally:
        for handle in prediction_cache_append_handles.values():
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        _shutdown_logger(logger)


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
    parser.add_argument("--run-mode", default=None, choices=["tuning", "testing"])
    args = parser.parse_args()

    config = load_replay_config(args.config)
    result = run_hmm_replay(
        config=config,
        run_mode=args.run_mode,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of_date=args.as_of_date,
        models=_parse_list_arg(args.models) or None,
        horizons=_parse_int_list_arg(args.horizons) or None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
