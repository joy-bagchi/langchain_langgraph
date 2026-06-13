from __future__ import annotations

from typing import Any

import pandas as pd


REGIME_ALIASES = {
    "STABLE_LOW_VOL": "STABLE_LOW_VOL_TREND",
    "STABLE_LOW_VOL_TREND": "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP": "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION": "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF": "HIGH_VOL_RISK_OFF",
}

REGIME_SEVERITY = {
    "STABLE_LOW_VOL_TREND": 0,
    "MID_VOL_CHOP": 1,
    "VOL_EXPANSION_TRANSITION": 2,
    "HIGH_VOL_RISK_OFF": 3,
}

LOW_RISK_STATES = {"STABLE_LOW_VOL_TREND", "MID_VOL_CHOP"}
HIGHER_RISK_STATES = {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"}


def normalize_regime_label(value: str) -> str:
    key = str(value or "").strip().upper()
    return REGIME_ALIASES.get(key, key)


def regime_severity(value: str) -> int | None:
    return REGIME_SEVERITY.get(normalize_regime_label(value))


def regime_risk_bucket(value: str) -> str:
    normalized = normalize_regime_label(value)
    return "HIGHER_VOL_RISK" if normalized in HIGHER_RISK_STATES else "LOW_RISK"


def _direction_label(change: float | None, *, flat_threshold: float) -> str | None:
    if change is None:
        return None
    value = float(change)
    if abs(value) < float(flat_threshold):
        return "FLAT"
    return "UP" if value > 0.0 else "DOWN"


def _expected_direction_set(predicted_state: str) -> set[str]:
    normalized = normalize_regime_label(predicted_state)
    if normalized in {"STABLE_LOW_VOL_TREND", "MID_VOL_CHOP"}:
        return {"DOWN", "FLAT"}
    if normalized in {"VOL_EXPANSION_TRANSITION", "HIGH_VOL_RISK_OFF"}:
        return {"UP", "FLAT"}
    return {"UP", "DOWN", "FLAT"}


def _directional_correct(observed: str | None, expected: set[str]) -> bool | None:
    if observed is None:
        return None
    return observed in expected


def score_prediction(prediction: dict[str, Any], outcome: dict[str, Any], *, horizon: int) -> dict[str, Any]:
    key = f"{horizon}d"
    realized_label = outcome.get(f"realized_regime_label_{key}")
    if realized_label is None:
        return {
            "as_of_date": prediction["as_of_date"],
            "model_name": prediction["model_name"],
            "horizon": horizon,
            "score_available": False,
        }

    predicted_state = normalize_regime_label(str(prediction["top_state"]))
    realized_state = normalize_regime_label(str(realized_label))
    predicted_index = regime_severity(predicted_state)
    realized_index = regime_severity(realized_state)
    severity_gap = (
        abs(int(predicted_index) - int(realized_index))
        if predicted_index is not None and realized_index is not None
        else None
    )

    vol_event = bool(outcome.get(f"vix_rose_{key}", False) or outcome.get(f"rv_expanded_{key}", False))
    spike_event = bool(outcome.get(f"vix_spike_{key}", False))
    predicted_vol_prob = float(prediction["transition_probabilities"].get(f"to_higher_vol_{key}", 0.0))
    predicted_spike_prob = min(1.0, max(0.0, predicted_vol_prob))

    posture = str(dict(prediction.get("policy_output", {})).get("overwrite_posture", "NO_OVERWRITE"))
    posture_consistent = (
        (posture in {"NO_OVERWRITE", "LIGHT_OVERWRITE"} and not vol_event)
        or (posture in {"MEDIUM_OVERWRITE", "AGGRESSIVE_OVERWRITE"} and vol_event)
    )

    predicted_risk_bucket = regime_risk_bucket(predicted_state)
    actual_risk_bucket = regime_risk_bucket(realized_state)
    false_alarm = bool(predicted_risk_bucket == "HIGHER_VOL_RISK" and actual_risk_bucket == "LOW_RISK")
    missed_risk = bool(predicted_risk_bucket == "LOW_RISK" and actual_risk_bucket == "HIGHER_VOL_RISK")

    vix_direction = _direction_label(outcome.get(f"vix_change_{key}"), flat_threshold=0.01)
    vvix_direction = _direction_label(outcome.get(f"vvix_change_{key}"), flat_threshold=0.01)
    rv21_asof = outcome.get("rv21_asof")
    rv21_change = outcome.get(f"rv21_change_{key}")
    rv21_pct_change = None
    if rv21_change is not None and rv21_asof not in {None, 0, 0.0}:
        rv21_pct_change = float(rv21_change) / float(rv21_asof)
    rv_direction = _direction_label(rv21_pct_change, flat_threshold=0.01)
    spy_direction = _direction_label(outcome.get(f"spy_return_{key}"), flat_threshold=0.0015)

    expected_direction = _expected_direction_set(predicted_state)
    directional_vix_correct = _directional_correct(vix_direction, expected_direction)
    directional_vvix_correct = _directional_correct(vvix_direction, expected_direction)
    directional_rv_correct = _directional_correct(rv_direction, expected_direction)
    directional_values = [
        item
        for item in (directional_vix_correct, directional_vvix_correct, directional_rv_correct)
        if item is not None
    ]
    combined_directional = (
        float(sum(1.0 if bool(item) else 0.0 for item in directional_values) / len(directional_values))
        if directional_values
        else None
    )

    return {
        "as_of_date": prediction["as_of_date"],
        "model_name": prediction["model_name"],
        "horizon": horizon,
        "score_available": True,
        "predicted_state": predicted_state,
        "realized_state": realized_state,
        "state_match": bool(predicted_state == realized_state),
        "predicted_severity_index": predicted_index,
        "realized_severity_index": realized_index,
        "severity_gap": severity_gap,
        "adjacent_correct": bool(severity_gap is not None and severity_gap <= 1),
        "severe_miss": bool(severity_gap is not None and severity_gap >= 2),
        "predicted_risk_bucket": predicted_risk_bucket,
        "actual_risk_bucket": actual_risk_bucket,
        "risk_bucket_correct": bool(predicted_risk_bucket == actual_risk_bucket),
        "false_alarm": false_alarm,
        "missed_risk": missed_risk,
        "vol_event": vol_event,
        "predicted_vol_prob": predicted_vol_prob,
        "brier_vol_expansion": float((predicted_vol_prob - float(vol_event)) ** 2),
        "vix_spike_event": spike_event,
        "predicted_vix_spike_prob": predicted_spike_prob,
        "brier_vix_spike": float((predicted_spike_prob - float(spike_event)) ** 2),
        "vix_direction": vix_direction,
        "vvix_direction": vvix_direction,
        "rv_direction": rv_direction,
        "spy_direction": spy_direction,
        "directional_vix_correct": directional_vix_correct,
        "directional_vvix_correct": directional_vvix_correct,
        "directional_rv_correct": directional_rv_correct,
        "combined_vol_directional_accuracy": combined_directional,
        "posture": posture,
        "posture_consistent": bool(posture_consistent),
    }


def summarize_scores(scored_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not scored_records:
        return pd.DataFrame(
            columns=[
                "model_name",
                "horizon",
                "accuracy",
                "adjacent_tolerant_accuracy",
                "severe_miss_rate",
                "brier_vol_expansion",
                "brier_vix_spike",
                "risk_bucket_accuracy",
                "false_alarm_rate",
                "missed_risk_rate",
                "vix_directional_accuracy",
                "vvix_directional_accuracy",
                "rv_directional_accuracy",
                "combined_vol_directional_accuracy",
                "avg_lead_quality",
                "notes",
            ]
        )
    frame = pd.DataFrame(scored_records)
    frame = frame[frame["score_available"] == True]  # noqa: E712
    grouped = frame.groupby(["model_name", "horizon"], as_index=False).agg(
        accuracy=("state_match", "mean"),
        adjacent_tolerant_accuracy=("adjacent_correct", "mean"),
        severe_miss_rate=("severe_miss", "mean"),
        brier_vol_expansion=("brier_vol_expansion", "mean"),
        brier_vix_spike=("brier_vix_spike", "mean"),
        risk_bucket_accuracy=("risk_bucket_correct", "mean"),
        false_alarm_rate=("false_alarm", "mean"),
        missed_risk_rate=("missed_risk", "mean"),
        vix_directional_accuracy=("directional_vix_correct", "mean"),
        vvix_directional_accuracy=("directional_vvix_correct", "mean"),
        rv_directional_accuracy=("directional_rv_correct", "mean"),
        combined_vol_directional_accuracy=("combined_vol_directional_accuracy", "mean"),
        avg_lead_quality=("posture_consistent", "mean"),
    )
    grouped["notes"] = ""
    return grouped


def build_prediction_distribution(scored_records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(scored_records)
    if frame.empty:
        return pd.DataFrame(columns=["model_name", "horizon", "predicted_state", "count", "percent"])
    scoped = frame[frame["score_available"] == True]  # noqa: E712
    grouped = (
        scoped.groupby(["model_name", "horizon", "predicted_state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = grouped.groupby(["model_name", "horizon"])["count"].transform("sum")
    grouped["percent"] = grouped["count"] / totals.replace({0: 1})
    return grouped.sort_values(["model_name", "horizon", "predicted_state"]).reset_index(drop=True)


def build_outcome_distribution(scored_records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(scored_records)
    if frame.empty:
        return pd.DataFrame(columns=["horizon", "realized_state", "count", "percent"])
    scoped = frame[frame["score_available"] == True].drop_duplicates(subset=["as_of_date", "horizon"])  # noqa: E712
    grouped = (
        scoped.groupby(["horizon", "realized_state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = grouped.groupby(["horizon"])["count"].transform("sum")
    grouped["percent"] = grouped["count"] / totals.replace({0: 1})
    return grouped.sort_values(["horizon", "realized_state"]).reset_index(drop=True)


def build_confusion_matrix_by_horizon(scored_records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(scored_records)
    if frame.empty:
        return pd.DataFrame(
            columns=["model_name", "horizon", "predicted_state", "realized_state", "count", "row_percent"]
        )
    scoped = frame[frame["score_available"] == True]  # noqa: E712
    grouped = (
        scoped.groupby(["model_name", "horizon", "predicted_state", "realized_state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    row_totals = grouped.groupby(["model_name", "horizon", "predicted_state"])["count"].transform("sum")
    grouped["row_percent"] = grouped["count"] / row_totals.replace({0: 1})
    return grouped.sort_values(["model_name", "horizon", "predicted_state", "realized_state"]).reset_index(drop=True)


def build_false_alarm_and_missed_risk_reports(
    *,
    scored_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not scored_records:
        empty = pd.DataFrame(
            columns=[
                "as_of_date",
                "model_name",
                "horizon",
                "predicted_state",
                "realized_state",
                "vix_change_pct",
                "vvix_change_pct",
                "rv21_change",
                "spy_return_pct",
                "vix",
                "vvix_vix_ratio",
                "term_structure_slope",
                "avg_pairwise_corr_21d",
                "first_eigenvalue_share_21d",
                "effective_rank_21d",
                "log_det_corr_21d",
            ]
        )
        return empty, empty
    pred_map = {(row["as_of_date"], row["model_name"]): row for row in predictions}
    out_map = {row["as_of_date"]: row for row in outcomes}
    rows: list[dict[str, Any]] = []
    for scored in scored_records:
        if not scored.get("score_available"):
            continue
        as_of = str(scored["as_of_date"])
        model_name = str(scored["model_name"])
        horizon = int(scored["horizon"])
        key = f"{horizon}d"
        prediction = pred_map.get((as_of, model_name), {})
        outcome = out_map.get(as_of, {})
        feature_snapshot = dict(prediction.get("feature_snapshot", {}))
        rows.append(
            {
                "as_of_date": as_of,
                "model_name": model_name,
                "horizon": horizon,
                "predicted_state": scored.get("predicted_state"),
                "realized_state": scored.get("realized_state"),
                "false_alarm": bool(scored.get("false_alarm", False)),
                "missed_risk": bool(scored.get("missed_risk", False)),
                "vix_direction": scored.get("vix_direction"),
                "rv_direction": scored.get("rv_direction"),
                "vix_change_pct": outcome.get(f"vix_change_{key}"),
                "vvix_change_pct": outcome.get(f"vvix_change_{key}"),
                "rv21_change": outcome.get(f"rv21_change_{key}"),
                "spy_return_pct": outcome.get(f"spy_return_{key}"),
                "vix": feature_snapshot.get("vix"),
                "vvix_vix_ratio": feature_snapshot.get("vvix_vix_ratio"),
                "term_structure_slope": feature_snapshot.get("term_structure_slope"),
                "avg_pairwise_corr_21d": feature_snapshot.get("avg_pairwise_corr_21d"),
                "first_eigenvalue_share_21d": feature_snapshot.get("first_eigenvalue_share_21d"),
                "effective_rank_21d": feature_snapshot.get("effective_rank_21d"),
                "log_det_corr_21d": feature_snapshot.get("log_det_corr_21d"),
            }
        )
    frame = pd.DataFrame(rows)
    false_alarms = frame[
        (frame["false_alarm"] == True)  # noqa: E712
        & (frame["vix_direction"].isin(["DOWN", "FLAT"]))
    ].reset_index(drop=True)
    missed_risks = frame[
        (frame["missed_risk"] == True)  # noqa: E712
        & ((frame["vix_direction"] == "UP") | (frame["rv_direction"] == "UP"))
    ].reset_index(drop=True)
    drop_cols = ["false_alarm", "missed_risk", "vix_direction", "rv_direction"]
    return false_alarms.drop(columns=drop_cols, errors="ignore"), missed_risks.drop(columns=drop_cols, errors="ignore")


def _disagreement_type(v3_state: str, comparison_state: str) -> str:
    v3_sev = regime_severity(v3_state)
    comparison_sev = regime_severity(comparison_state)
    if v3_sev is not None and comparison_sev is not None:
        if v3_sev < comparison_sev:
            return "v3_downgrade"
        if v3_sev > comparison_sev:
            return "v3_upgrade"
    return "v3_same_bucket_different_state"


def build_disagreement_attribution(
    *,
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    scored_records: list[dict[str, Any]],
    horizons: list[int],
) -> pd.DataFrame:
    prediction_map = {
        (str(row["as_of_date"]), str(row["model_name"])): row
        for row in predictions
    }
    outcome_map = {str(row["as_of_date"]): row for row in outcomes}
    score_map = {
        (str(row["as_of_date"]), str(row["model_name"]), int(row["horizon"])): row
        for row in scored_records
        if bool(row.get("score_available", False))
    }
    comparison_models = ("heuristic", "hmm_v1_core", "hmm_v2_core_plus_sector_corr")
    all_dates = sorted({str(row["as_of_date"]) for row in predictions})
    rows: list[dict[str, Any]] = []
    for as_of_date in all_dates:
        v3 = prediction_map.get((as_of_date, "hmm_v3_core_plus_sector_geometry"))
        if not v3:
            continue
        v3_state = normalize_regime_label(str(v3.get("top_state", "")))
        v3_severity = regime_severity(v3_state)
        v3_bucket = regime_risk_bucket(v3_state)
        feature_snapshot = dict(v3.get("feature_snapshot", {}))
        warning_items = [
            feature
            for feature in (
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
            )
            if feature_snapshot.get(feature) is None
        ]
        outcome = outcome_map.get(as_of_date, {})
        for comparison_model in comparison_models:
            comparison = prediction_map.get((as_of_date, comparison_model))
            if not comparison:
                continue
            comparison_state = normalize_regime_label(str(comparison.get("top_state", "")))
            if comparison_state == v3_state:
                continue
            comparison_severity = regime_severity(comparison_state)
            comparison_bucket = regime_risk_bucket(comparison_state)
            disagreement_type = _disagreement_type(v3_state, comparison_state)
            severity_delta = (
                int(v3_severity) - int(comparison_severity)
                if v3_severity is not None and comparison_severity is not None
                else None
            )
            is_opposite_bucket = bool(v3_bucket != comparison_bucket)
            row: dict[str, Any] = {
                "as_of_date": as_of_date,
                "comparison_model": comparison_model,
                "comparison_state": comparison_state,
                "hmm_v3_state": v3_state,
                "disagreement_type": disagreement_type,
                "is_opposite_bucket": is_opposite_bucket,
                "severity_delta": severity_delta,
                "vix": feature_snapshot.get("vix"),
                "vvix": feature_snapshot.get("vvix"),
                "vvix_vix_ratio": feature_snapshot.get("vvix_vix_ratio"),
                "vvix_vix_z_22d": feature_snapshot.get("vvix_vix_z_22d"),
                "vix_vix3m_ratio": feature_snapshot.get("vix_vix3m_ratio"),
                "vix9d_vix_ratio": feature_snapshot.get("vix9d_vix_ratio"),
                "term_structure_slope": feature_snapshot.get("term_structure_slope"),
                "realized_vol_5d": feature_snapshot.get("realized_vol_5d"),
                "realized_vol_21d": feature_snapshot.get("realized_vol_21d"),
                "realized_vol_trend": (
                    (feature_snapshot.get("realized_vol_5d") or 0.0) - (feature_snapshot.get("realized_vol_21d") or 0.0)
                    if feature_snapshot.get("realized_vol_5d") is not None and feature_snapshot.get("realized_vol_21d") is not None
                    else None
                ),
                "spy_return_1d": feature_snapshot.get("spy_return_1d"),
                "drawdown_21d": feature_snapshot.get("drawdown_21d"),
                "trend_persistence_21d": feature_snapshot.get("trend_persistence_21d"),
                "avg_pairwise_corr_21d": feature_snapshot.get("avg_pairwise_corr_21d"),
                "first_eigenvalue_share_21d": feature_snapshot.get("first_eigenvalue_share_21d"),
                "effective_rank_21d": feature_snapshot.get("effective_rank_21d"),
                "log_det_corr_21d": feature_snapshot.get("log_det_corr_21d"),
                "warning": "missing features: " + ", ".join(warning_items) if warning_items else "",
            }
            win_counter = 0
            loss_counter = 0
            for horizon in horizons:
                suffix = f"{horizon}d"
                v3_score = score_map.get((as_of_date, "hmm_v3_core_plus_sector_geometry", int(horizon)), {})
                comparison_score = score_map.get((as_of_date, comparison_model, int(horizon)), {})
                realized_state = normalize_regime_label(str(outcome.get(f"realized_regime_label_{suffix}", "")))
                realized_severity = regime_severity(realized_state)
                realized_bucket = regime_risk_bucket(realized_state) if realized_state else "LOW_RISK"
                v3_distance = (
                    abs(int(v3_severity) - int(realized_severity))
                    if v3_severity is not None and realized_severity is not None
                    else None
                )
                comparison_distance = (
                    abs(int(comparison_severity) - int(realized_severity))
                    if comparison_severity is not None and realized_severity is not None
                    else None
                )
                if v3_distance is None or comparison_distance is None:
                    v3_won = "tie"
                elif v3_distance < comparison_distance:
                    v3_won = "true"
                    win_counter += 1
                elif v3_distance > comparison_distance:
                    v3_won = "false"
                    loss_counter += 1
                else:
                    v3_won = "tie"
                v3_bucket_correct = bool(v3_score.get("risk_bucket_correct", False))
                comparison_bucket_correct = bool(comparison_score.get("risk_bucket_correct", False))
                if v3_bucket_correct and not comparison_bucket_correct:
                    v3_bucket_won = "true"
                elif comparison_bucket_correct and not v3_bucket_correct:
                    v3_bucket_won = "false"
                else:
                    v3_bucket_won = "tie"
                row.update(
                    {
                        f"realized_state_{suffix}": realized_state,
                        f"realized_risk_bucket_{suffix}": realized_bucket,
                        f"spy_return_{suffix}": outcome.get(f"spy_return_{suffix}"),
                        f"vix_change_pct_{suffix}": outcome.get(f"vix_change_{suffix}"),
                        f"vvix_change_pct_{suffix}": outcome.get(f"vvix_change_{suffix}"),
                        f"rv21_change_{suffix}": outcome.get(f"rv21_change_{suffix}"),
                        f"vix_spike_{suffix}": outcome.get(f"vix_spike_{suffix}"),
                        f"rv_expanded_{suffix}": outcome.get(f"rv_expanded_{suffix}"),
                        f"risk_bucket_correct_hmm_v3_{suffix}": v3_bucket_correct,
                        f"risk_bucket_correct_comparison_{suffix}": comparison_bucket_correct,
                        f"exact_correct_hmm_v3_{suffix}": bool(v3_score.get("state_match", False)),
                        f"exact_correct_comparison_{suffix}": bool(comparison_score.get("state_match", False)),
                        f"directional_vol_correct_hmm_v3_{suffix}": v3_score.get("combined_vol_directional_accuracy"),
                        f"directional_vol_correct_comparison_{suffix}": comparison_score.get("combined_vol_directional_accuracy"),
                        f"v3_won_{suffix}": v3_won,
                        f"v3_bucket_won_{suffix}": v3_bucket_won,
                    }
                )
            if win_counter > loss_counter:
                row["v3_result"] = "win"
            elif loss_counter > win_counter:
                row["v3_result"] = "loss"
            else:
                row["v3_result"] = "tie"
            rows.append(row)
    return pd.DataFrame(rows)


def build_disagreement_summary(disagreement_df: pd.DataFrame, *, horizons: list[int]) -> pd.DataFrame:
    if disagreement_df.empty:
        return pd.DataFrame(
            columns=[
                "comparison_model",
                "horizon",
                "total_disagreements",
                "v3_win_rate",
                "v3_loss_rate",
                "tie_rate",
                "v3_bucket_win_rate",
                "v3_bucket_loss_rate",
                "downgrade_count",
                "upgrade_count",
                "downgrade_success_rate",
                "upgrade_success_rate",
                "false_suppression_rate",
                "false_alarm_filter_success_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    for comparison_model in sorted(disagreement_df["comparison_model"].unique()):
        scoped_model = disagreement_df[disagreement_df["comparison_model"] == comparison_model]
        for horizon in horizons:
            suffix = f"{horizon}d"
            won = scoped_model[f"v3_won_{suffix}"].astype(str)
            bucket = scoped_model[f"v3_bucket_won_{suffix}"].astype(str)
            downgrade = scoped_model["disagreement_type"] == "v3_downgrade"
            upgrade = scoped_model["disagreement_type"] == "v3_upgrade"
            realized_high = scoped_model[f"realized_risk_bucket_{suffix}"] == "HIGHER_VOL_RISK"
            realized_low = scoped_model[f"realized_risk_bucket_{suffix}"] == "LOW_RISK"
            vix_down_flat = scoped_model[f"vix_change_pct_{suffix}"].fillna(0.0).abs() < 0.01
            vix_down_flat = vix_down_flat | (scoped_model[f"vix_change_pct_{suffix}"].fillna(0.0) < 0.0)
            opposite_bucket = scoped_model["disagreement_type"] == "v3_opposite_bucket"
            if "is_opposite_bucket" in scoped_model.columns:
                opposite_bucket = scoped_model["is_opposite_bucket"] == True  # noqa: E712
            false_suppression = downgrade & opposite_bucket & realized_high
            false_alarm_filter = downgrade & opposite_bucket & realized_low & vix_down_flat
            downgrade_success = downgrade & (won == "true")
            upgrade_success = upgrade & (won == "true")
            count = len(scoped_model)
            rows.append(
                {
                    "comparison_model": comparison_model,
                    "horizon": int(horizon),
                    "total_disagreements": int(count),
                    "v3_win_rate": float((won == "true").mean() if count else 0.0),
                    "v3_loss_rate": float((won == "false").mean() if count else 0.0),
                    "tie_rate": float((won == "tie").mean() if count else 0.0),
                    "v3_bucket_win_rate": float((bucket == "true").mean() if count else 0.0),
                    "v3_bucket_loss_rate": float((bucket == "false").mean() if count else 0.0),
                    "downgrade_count": int(downgrade.sum()),
                    "upgrade_count": int(upgrade.sum()),
                    "downgrade_success_rate": float(downgrade_success.mean() if count else 0.0),
                    "upgrade_success_rate": float(upgrade_success.mean() if count else 0.0),
                    "false_suppression_rate": float(false_suppression.mean() if count else 0.0),
                    "false_alarm_filter_success_rate": float(false_alarm_filter.mean() if count else 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_geometry_case_files(disagreement_df: pd.DataFrame, *, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if disagreement_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    override = disagreement_df[
        disagreement_df["disagreement_type"].isin(["v3_downgrade", "v3_upgrade"])
        & disagreement_df["severity_delta"].abs().ge(1)
        & disagreement_df["comparison_model"].isin(["heuristic", "hmm_v1_core"])
    ].copy()
    win_cols = [f"v3_won_{h}d" for h in horizons]
    bucket_cols = [f"v3_bucket_won_{h}d" for h in horizons]
    success = disagreement_df[
        disagreement_df[win_cols].isin(["true"]).any(axis=1)
        | disagreement_df[bucket_cols].isin(["true"]).any(axis=1)
    ].copy()
    first_win_horizon: list[str] = []
    reason: list[str] = []
    for _, row in success.iterrows():
        chosen = ""
        why = ""
        for h in horizons:
            if str(row.get(f"v3_bucket_won_{h}d", "")) == "true":
                chosen = f"T+{h}"
                why = "risk_bucket"
                break
            if str(row.get(f"v3_won_{h}d", "")) == "true":
                chosen = f"T+{h}"
                why = "severity_distance"
                break
        first_win_horizon.append(chosen)
        reason.append(why)
    success["win_horizon"] = first_win_horizon
    success["why"] = reason
    false_supp_mask = disagreement_df["disagreement_type"].eq("v3_downgrade") & disagreement_df["comparison_model"].isin(
        ["heuristic", "hmm_v1_core", "hmm_v2_core_plus_sector_corr"]
    )
    if "is_opposite_bucket" in disagreement_df.columns:
        false_supp_mask = false_supp_mask & disagreement_df["is_opposite_bucket"].eq(True)  # noqa: E712
    any_high = pd.Series(False, index=disagreement_df.index)
    for h in horizons:
        suffix = f"{h}d"
        any_high = any_high | disagreement_df[f"realized_risk_bucket_{suffix}"].eq("HIGHER_VOL_RISK")
    false_supp_mask = false_supp_mask & any_high
    false_suppression = disagreement_df[false_supp_mask].copy()
    return override, false_suppression, success


def build_geometry_false_suppression_analysis(
    false_suppression_df: pd.DataFrame,
    *,
    horizons: list[int],
) -> pd.DataFrame:
    columns = ["segment", "metric", "category", "value"]
    if false_suppression_df.empty:
        return pd.DataFrame(
            [
                {
                    "segment": "overall",
                    "metric": "case_count",
                    "category": "",
                    "value": 0.0,
                }
            ],
            columns=columns,
        )

    frame = false_suppression_df.copy()
    if "v3_result" not in frame.columns:
        frame["v3_result"] = "tie"
    frame["v3_result"] = frame["v3_result"].astype(str).str.lower()

    segments: list[tuple[str, pd.DataFrame]] = [("overall", frame)]
    segments.append(("wins", frame[frame["v3_result"] == "win"]))
    segments.append(("losses", frame[frame["v3_result"] == "loss"]))

    rows: list[dict[str, Any]] = []

    def _add_metric(segment: str, metric: str, category: str, value: float) -> None:
        rows.append(
            {
                "segment": segment,
                "metric": metric,
                "category": category,
                "value": float(value),
            }
        )

    for segment_name, segment_df in segments:
        count = int(len(segment_df))
        _add_metric(segment_name, "case_count", "", float(count))
        if count == 0:
            for metric in (
                "avg_vix",
                "avg_vvix",
                "avg_vvix_vix_ratio",
                "avg_geometry_stress_score",
                "avg_downgrade_levels",
            ):
                _add_metric(segment_name, metric, "", 0.0)
            continue

        for horizon in horizons:
            outcome_col = f"realized_state_{horizon}d"
            if outcome_col in segment_df.columns:
                for label, value in segment_df[outcome_col].astype(str).value_counts().items():
                    _add_metric(segment_name, "count_by_realized_outcome", f"T+{horizon}:{label}", float(value))
            bucket_col = f"realized_risk_bucket_{horizon}d"
            if bucket_col in segment_df.columns:
                for label, value in segment_df[bucket_col].astype(str).value_counts().items():
                    _add_metric(segment_name, "count_by_realized_risk_bucket", f"T+{horizon}:{label}", float(value))

        _add_metric(segment_name, "avg_vix", "", float(segment_df["vix"].mean()) if "vix" in segment_df.columns else 0.0)
        _add_metric(segment_name, "avg_vvix", "", float(segment_df["vvix"].mean()) if "vvix" in segment_df.columns else 0.0)
        _add_metric(
            segment_name,
            "avg_vvix_vix_ratio",
            "",
            float(segment_df["vvix_vix_ratio"].mean()) if "vvix_vix_ratio" in segment_df.columns else 0.0,
        )
        _add_metric(
            segment_name,
            "avg_geometry_stress_score",
            "",
            float(segment_df["geometry_stress_score"].mean()) if "geometry_stress_score" in segment_df.columns else 0.0,
        )
        _add_metric(
            segment_name,
            "avg_downgrade_levels",
            "",
            float(segment_df["downgrade_levels"].mean()) if "downgrade_levels" in segment_df.columns else 0.0,
        )

    return pd.DataFrame(rows, columns=columns)
