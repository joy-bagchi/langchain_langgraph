"""Streamlit frontend for the volatility regime application."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_streamlit_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "agentic_vol_regime_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_streamlit_imports()

from agentic_vol_regime_app import (  # noqa: E402
    build_backtest_feature_store,
    default_agent_path,
    default_hmm_agent_path,
    default_hmm_v2_agent_path,
    default_hmm_v3_agent_path,
    default_hmm_v3_1_agent_path,
    default_ibkr_agent_path,
    default_ml_agent_path,
    load_or_run_historical_belief_report,
    load_latest_live_daily_observation,
    load_recent_hmm_state_history,
    reset_hmm_persisted_state,
    snapshot_hmm_baseline,
    resume_daily_regime_run,
    run_daily_regime_agent,
    run_hmm_replay_backtester,
    run_policy_backtester,
    run_ibkr_market_data_agent,
)
from agentic_vol_regime_app.config import AppPaths, load_json  # noqa: E402


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install it with `pip install streamlit` "
            "or `pip install .[ui]` from agentic_vol_regime_app."
        ) from exc
    return st


APP_PATHS = AppPaths.default()
DEFAULT_DAILY_INPUT = APP_PATHS.sample_inputs_dir / "daily_snapshot_watch.json"
DEFAULT_DAILY_LIVE_INPUT = APP_PATHS.sample_inputs_dir / "daily_snapshot_ibkr_live.json"
DEFAULT_IBKR_INPUT = APP_PATHS.sample_inputs_dir / "ibkr_spy_snapshot.json"
DEFAULT_BACKTEST_CONFIG = APP_PATHS.configs_dir / "backtest" / "hmm_replay_10y_hmmv4.yaml"
DEFAULT_POLICY_BACKTEST_CONFIG = APP_PATHS.configs_dir / "backtest" / "policy_backtest.yaml"
CARD_LABEL_COLOR = "#facc15"
CARD_VALUE_COLOR = "#fef3c7"
WIDGET_BORDER_COLOR = "#facc15"
WIDGET_BORDER_FOCUS = "#fde68a"
REGIME_ENGINE_OPTIONS = [
    "Heuristic Agent",
    "ML Agent",
    "HMMv1 Agent",
    "HMMv2 Agent",
    "HMMv3 Agent",
    "HMMv3.1 Meta-Blend Agent",
]
BACKTEST_MODEL_OPTIONS = [
    "heuristic",
    "hmm_v1_core",
    "hmm_v2_core_plus_sector_corr",
    "hmm_v3_core_plus_sector_geometry",
    "hmm_v3_1_meta_blend",
    "hmm_v4_path_aware_meta",
]
BACKTEST_HORIZON_OPTIONS = [1, 2, 3, 5, 10]
POLICY_MODEL_OPTIONS = [
    "hmm_v3_1_meta_blend",
    "hmm_v4_path_aware_meta",
]
TUNING_DEFAULT_LOOKBACK_DAYS = 756
TUNING_DEFAULT_MIN_TRAIN_ROWS = 504
TUNING_DEFAULT_MAX_REPLAY_DATES = 75
TUNING_DEFAULT_WINDOW_BDAYS = 90


def _default_backtest_models_for_run_mode(run_mode: str) -> list[str]:
    if str(run_mode).strip().lower() == "tuning":
        return ["hmm_v4_path_aware_meta", "hmm_v3_1_meta_blend"]
    return list(BACKTEST_MODEL_OPTIONS)


def _default_backtest_horizons_for_run_mode(run_mode: str) -> list[int]:
    if str(run_mode).strip().lower() == "tuning":
        return [1, 3]
    return [1, 2, 3, 5, 10]


def _compact_backtest_session_payload(result: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "report_path",
        "compact_summary_path",
        "run_log_path",
        "run_mode",
        "peak_rss_mb",
        "slowest_dates",
        "runtime_profile",
        "cache_hits",
        "cache_misses",
        "total_prediction_records",
        "total_scored_records",
        "summary_metrics_path",
        "economic_summary_path",
        "prediction_distribution_path",
        "outcome_distribution_path",
        "confusion_matrix_path",
        "false_alarms_path",
        "missed_risks_path",
        "top_feature_importances_path",
        "prediction_records_path",
        "outcome_records_path",
        "scored_records_path",
        "summary_metrics",
    ]
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in result:
            compact[key] = result.get(key)
    return compact


def _build_run_mode_preset_rows(
    *,
    selected_replay_run_mode: str,
    replay_scope: str,
    selected_models: list[str],
    selected_horizons: list[int],
    config_train_lookback_days: int,
    config_min_train_rows: int,
) -> list[dict[str, str]]:
    mode = str(selected_replay_run_mode).strip().lower()
    if mode == "tuning":
        target_window = f"Last {TUNING_DEFAULT_WINDOW_BDAYS} business days (cap {TUNING_DEFAULT_MAX_REPLAY_DATES} replay dates)"
        lookback = str(TUNING_DEFAULT_LOOKBACK_DAYS)
        min_rows = str(TUNING_DEFAULT_MIN_TRAIN_ROWS)
        max_dates = str(TUNING_DEFAULT_MAX_REPLAY_DATES)
        diagnostics = "Compact diagnostics (fast iteration)"
    else:
        target_window = "Full selected replay window"
        lookback = str(int(config_train_lookback_days))
        min_rows = str(int(config_min_train_rows))
        max_dates = "none"
        diagnostics = "Full diagnostics (final validation)"
    if replay_scope == "Single Date":
        target_window = "Single as-of replay date"
        max_dates = "1"
    return [
        {"Setting": "Run Mode", "Effective Value": mode},
        {"Setting": "Replay Scope", "Effective Value": str(replay_scope)},
        {"Setting": "Date Window Intent", "Effective Value": target_window},
        {"Setting": "Train Lookback Days", "Effective Value": lookback},
        {"Setting": "Min Train Rows", "Effective Value": min_rows},
        {"Setting": "Max Replay Dates", "Effective Value": max_dates},
        {"Setting": "Horizons", "Effective Value": ", ".join(str(item) for item in selected_horizons)},
        {"Setting": "Models", "Effective Value": ", ".join(selected_models)},
        {"Setting": "Diagnostics", "Effective Value": diagnostics},
    ]


def _pretty_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_json_text(value: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not value.strip():
        return dict(fallback)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a top-level JSON object.")
    return parsed


def _resolve_daily_agent_path_from_choice(choice: str) -> Path:
    if choice == "Heuristic Agent":
        return default_agent_path()
    if choice == "ML Agent":
        return default_ml_agent_path()
    if choice == "HMMv1 Agent":
        return default_hmm_agent_path()
    if choice == "HMMv2 Agent":
        return default_hmm_v2_agent_path()
    if choice == "HMMv3 Agent":
        return default_hmm_v3_agent_path()
    if choice == "HMMv3.1 Meta-Blend Agent":
        return default_hmm_v3_1_agent_path()
    return default_agent_path()


def _default_history_days_for_agent_choice(choice: str, fallback: int) -> int:
    return (
        756
        if choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent", "HMMv3.1 Meta-Blend Agent"}
        else int(fallback)
    )


def _render_runtime_diagnostics(st, result: dict[str, Any]) -> None:
    status = str(result.get("status", "unknown"))
    run_id = str(result.get("run_id", ""))
    current_step = result.get("current_step")
    last_error = result.get("last_error")
    pending_review = result.get("pending_review") or {}

    st.subheader("Workflow Diagnostics")
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    diag_col1.metric("Status", status)
    diag_col2.metric("Run ID", run_id[:12] if run_id else "n/a")
    diag_col3.metric("Current Step", str(current_step or "none"))

    if status == "completed":
        st.success("Workflow completed successfully.")
    elif status == "awaiting_review":
        review_type = pending_review.get("review_type", "human_review")
        st.warning(f"Workflow is awaiting review. Review type: `{review_type}`.")
    elif status == "failed":
        st.error("Workflow failed.")
    else:
        st.info(f"Workflow status: `{status}`")

    if last_error:
        st.error(f"Last error: {last_error}")

    if pending_review:
        with st.expander("Pending Review Payload"):
            st.code(_pretty_json(pending_review), language="json")

    step_history = list(result.get("step_history", []))
    if step_history:
        latest_step = dict(step_history[-1])
        with st.expander("Latest Step Execution"):
            st.code(_pretty_json(latest_step), language="json")


def _render_summary_card(st, *, label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 0.75rem;
            padding: 0.85rem 0.95rem;
            min-height: 5.5rem;
            background: rgba(255, 255, 255, 0.02);
        ">
            <div style="
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: {CARD_LABEL_COLOR};
                margin-bottom: 0.45rem;
            ">{label}</div>
            <div style="
                font-size: 1.05rem;
                line-height: 1.25;
                font-weight: 600;
                word-break: break-word;
                color: {CARD_VALUE_COLOR};
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_widget_theme(st) -> None:
    st.markdown(
        f"""
        <style>
        div[data-baseweb="select"] > div {{
            border-color: {WIDGET_BORDER_COLOR} !important;
        }}

        div[data-baseweb="select"] > div:hover {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
        }}

        div[data-baseweb="select"] > div:focus-within {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
            box-shadow: 0 0 0 1px {WIDGET_BORDER_FOCUS} !important;
        }}

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
            border-color: {WIDGET_BORDER_COLOR} !important;
        }}

        .stTextInput input:hover,
        .stTextArea textarea:hover,
        .stNumberInput input:hover {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
        }}

        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
            box-shadow: 0 0 0 1px {WIDGET_BORDER_FOCUS} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tone_for_regime(regime: str) -> tuple[str, str]:
    value = regime.upper()
    if value == "HIGH_VOL_RISK_OFF":
        return ("#c62828", "#fdecea")
    if value in {"VOL_EXPANSION_TRANSITION", "MID_VOL_CHOP"}:
        return ("#a15c00", "#fff4db")
    return ("#2e7d32", "#edf7ed")


def _tone_for_alert(alert: str) -> tuple[str, str]:
    value = alert.upper()
    if value in {"HIGH_RISK", "CRITICAL"}:
        return ("#c62828", "#fdecea")
    if value in {"WARNING", "WATCH"}:
        return ("#a15c00", "#fff4db")
    return ("#2e7d32", "#edf7ed")


def _tone_for_action(action: str) -> tuple[str, str]:
    value = action.upper()
    if value in {"MANUAL_REVIEW", "AGGRESSIVE_OVERWRITE"}:
        return ("#c62828", "#fdecea")
    if value in {"MEDIUM_OVERWRITE", "LIGHT_OVERWRITE", "REDUCE_RISK"}:
        return ("#a15c00", "#fff4db")
    return ("#2e7d32", "#edf7ed")


def _render_color_summary_cards(st, *, regime: str, alert: str, action: str) -> None:
    regime_fg, regime_bg = _tone_for_regime(regime)
    alert_fg, alert_bg = _tone_for_alert(alert)
    action_fg, action_bg = _tone_for_action(action)
    st.subheader("Summary")
    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.85rem; margin-bottom: 1rem;">
            <div style="border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 0.85rem; padding: 0.9rem 1rem; background: rgba(255,255,255,0.02); min-height: 6rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: {CARD_LABEL_COLOR}; margin-bottom: 0.45rem;">Current Regime</div>
                <div><span style="background:{regime_bg}; color:{regime_fg}; padding:0.28rem 0.58rem; border-radius:999px; font-weight:700;">{regime}</span></div>
            </div>
            <div style="border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 0.85rem; padding: 0.9rem 1rem; background: rgba(255,255,255,0.02); min-height: 6rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: {CARD_LABEL_COLOR}; margin-bottom: 0.45rem;">Transition Risk</div>
                <div><span style="background:{alert_bg}; color:{alert_fg}; padding:0.28rem 0.58rem; border-radius:999px; font-weight:700;">{alert}</span></div>
            </div>
            <div style="border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 0.85rem; padding: 0.9rem 1rem; background: rgba(255,255,255,0.02); min-height: 6rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: {CARD_LABEL_COLOR}; margin-bottom: 0.45rem;">Recommended Posture</div>
                <div><span style="background:{action_bg}; color:{action_fg}; padding:0.28rem 0.58rem; border-radius:999px; font-weight:700;">{action}</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overwrite_plan(st, *, policy: dict[str, Any]) -> None:
    strike = policy.get("overwrite_call_strike")
    dte = policy.get("overwrite_dte")
    rationale = str(policy.get("overwrite_rationale") or "").strip()
    if strike is None or dte is None:
        return
    st.subheader("Overwrite Plan")
    plan_col1, plan_col2 = st.columns(2)
    _render_summary_card(plan_col1, label="Suggested Strike", value=f"{float(strike):.2f}")
    _render_summary_card(plan_col2, label="Suggested DTE", value=str(int(dte)))
    if rationale:
        st.caption(rationale)


def _render_hmm_summary(st, *, hmm_belief: dict[str, Any]) -> None:
    if not hmm_belief:
        return

    is_trained = bool(hmm_belief.get("is_trained", False))
    training_status = str(hmm_belief.get("training_status", "unknown"))
    state_probabilities = dict(hmm_belief.get("state_probabilities", {}))
    transition_probabilities = dict(hmm_belief.get("transition_probabilities", {}))
    training_row_count = int(hmm_belief.get("training_row_count", 0) or 0)
    configured_train_window = int(hmm_belief.get("configured_train_window", 0) or 0)

    st.subheader("HMM Summary")
    if not is_trained:
        st.warning(
            "HMM is not trained enough for this run. The app is showing the HMM warning state, not a learned regime posterior."
        )
    hmm_col1, hmm_col2, hmm_col3, hmm_col4 = st.columns(4)
    _render_summary_card(hmm_col1, label="Training Status", value=training_status)
    _render_summary_card(hmm_col2, label="Top HMM State", value=str(hmm_belief.get("top_state", "n/a")))
    _render_summary_card(
        hmm_col3,
        label="Expected Duration",
        value=f"{float(hmm_belief.get('current_state_expected_duration_days', 0.0)):.1f} days",
    )
    _render_summary_card(
        hmm_col4,
        label="5d Expansion/Stress",
        value=f"{float(transition_probabilities.get('to_vol_expansion_or_high_vol_5d', 0.0)):.2f}",
    )

    if state_probabilities:
        st.caption(
            f"Usable training rows: {training_row_count}"
            + (f" / {configured_train_window}" if configured_train_window > 0 else "")
        )
        rows = []
        for state, probability in state_probabilities.items():
            rows.append(
                {
                    "state": state,
                    "probability": float(probability),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_hmm_history(st, *, hmm_history: list[dict[str, Any]]) -> None:
    if not hmm_history:
        return
    with st.expander("HMM State History", expanded=False):
        rows = []
        for item in hmm_history:
            top_state = str(item.get("top_state", "n/a"))
            state_probabilities = dict(item.get("state_probabilities", {}))
            transition_probabilities = dict(item.get("transition_probabilities", {}))
            rows.append(
                {
                    "as_of": item.get("observation_as_of"),
                    "top_state": top_state,
                    "posterior": float(state_probabilities.get(top_state, 0.0)),
                    "expected_duration_days": float(item.get("current_state_expected_duration_days", 0.0) or 0.0),
                    "to_expansion_or_stress_5d": float(
                        transition_probabilities.get("to_vol_expansion_or_high_vol_5d", 0.0)
                    ),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)


def _extract_governance_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(result.get("step_history", [])):
        metadata = dict(item.get("metadata", {}))
        evaluation = dict(metadata.get("evaluation", {}))
        rows.append(
            {
                "step": str(item.get("step_id", "")),
                "status": str(item.get("status", "")),
                "guardrail": str(metadata.get("guardrail_action", "allow")),
                "evaluation": str(evaluation.get("action", "allow")),
                "score": evaluation.get("score"),
                "review_trigger": (
                    "guardrail"
                    if metadata.get("guardrail_action") == "escalate"
                    else "evaluation"
                    if metadata.get("evaluation_action") == "escalate"
                    else ""
                ),
                "notes": " | ".join(
                    [
                        *[str(reason) for reason in list(metadata.get("guardrail_reasons", []))],
                        *[str(finding) for finding in list(evaluation.get("findings", []))],
                    ]
                ),
            }
        )
    return rows


def _render_governance_panel(
    st,
    *,
    result: dict[str, Any],
    critic_review: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    pending_review = dict(result.get("pending_review") or {})
    review_decision = dict(result.get("named_outputs", {})).get("review_decision") or {}
    step_rows = _extract_governance_rows(result)

    st.subheader("Controls and Governance")

    critic_verdict = str(critic_review.get("verdict", "n/a"))
    policy_action = str(policy.get("recommended_action", "n/a"))
    review_type = str(pending_review.get("review_type", "none")) if pending_review else "none"
    human_decision = str(dict(review_decision).get("decision", "none")) if review_decision else "none"

    gov_col1, gov_col2, gov_col3, gov_col4 = st.columns(4)
    _render_summary_card(gov_col1, label="Critic Verdict", value=critic_verdict)
    _render_summary_card(gov_col2, label="Policy Action", value=policy_action)
    _render_summary_card(gov_col3, label="Pending Review", value=review_type)
    _render_summary_card(gov_col4, label="Human Decision", value=human_decision)

    findings = list(critic_review.get("findings", []))
    risk_notes = list(policy.get("risk_notes", []))
    rationale = list(policy.get("rationale", []))

    summary_lines: list[str] = []
    if findings:
        summary_lines.append("Critic: " + " | ".join(str(item) for item in findings))
    if rationale:
        summary_lines.append("Policy rationale: " + " | ".join(str(item) for item in rationale))
    if risk_notes:
        summary_lines.append("Risk notes: " + " | ".join(str(item) for item in risk_notes))
    if pending_review:
        summary_lines.append(
            "Pending review trigger: "
            + str(pending_review.get("review_type", "review"))
            + " | "
            + " | ".join(str(item) for item in pending_review.get("guardrail_reasons", []) or pending_review.get("evaluation_findings", []))
        )
    elif review_decision:
        review_notes = str(dict(review_decision).get("notes") or "").strip()
        summary_lines.append(
            "Review resolution: "
            + str(dict(review_decision).get("decision", "unknown"))
            + (f" | {review_notes}" if review_notes else "")
        )

    if summary_lines:
        st.caption("\n\n".join(summary_lines))

    if step_rows:
        st.dataframe(step_rows, width="stretch", hide_index=True)


def _strip_report_heading_and_summary(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("# Daily Volatility Regime Report"):
            continue
        if line.startswith("## Summary"):
            skip = True
            continue
        if skip and line.startswith("## ") and line.strip() != "## Summary":
            skip = False
        if skip:
            continue
        output.append(line)
    return "\n".join(output).strip()


def _render_daily_result(st, result: dict[str, Any], *, hmm_history: list[dict[str, Any]] | None = None) -> None:
    outputs = dict(result.get("named_outputs", {}))
    daily_report = dict(outputs.get("daily_report", {}))
    belief_state = dict(outputs.get("belief_state", {}))
    alert_record = dict(outputs.get("alert_record", {}))
    policy = dict(outputs.get("policy_recommendation", {}))
    critic_review = dict(outputs.get("critic_review", {}))
    hmm_belief = dict(outputs.get("hmm_belief", {}))
    agent_name = str(dict(result.get("agent", {})).get("name", "")).strip()

    _render_runtime_diagnostics(st, result)
    if agent_name:
        st.caption(f"Agent engine: {agent_name}")

    top_regime = (
        max(dict(belief_state.get("beliefs", {})), key=dict(belief_state.get("beliefs", {})).get)
        if belief_state.get("beliefs")
        else "n/a"
    )
    _render_color_summary_cards(
        st,
        regime=top_regime,
        alert=str(alert_record.get("severity", "n/a")),
        action=str(policy.get("recommended_action", "n/a")),
    )
    _render_overwrite_plan(st, policy=policy)
    is_hmm_run = bool(hmm_belief) and "hmm" in agent_name.lower()
    if is_hmm_run:
        _render_hmm_summary(st, hmm_belief=hmm_belief)
        _render_hmm_history(st, hmm_history=hmm_history or [])
    _render_governance_panel(st, result=result, critic_review=critic_review, policy=policy)

    if daily_report.get("markdown"):
        st.subheader("Daily Report")
        st.markdown(_strip_report_heading_and_summary(str(daily_report["markdown"])))

    st.subheader("Belief Vector")
    beliefs = dict(belief_state.get("beliefs", {}))
    if beliefs:
        st.dataframe(
            [{"regime": key, "probability": value} for key, value in beliefs.items()],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Internal Run State"):
        st.code(_pretty_json(result), language="json")


def _render_inline_review_actions(
    st,
    *,
    result: dict[str, Any],
    agent_path: Path,
    storage_root: str,
    database_url: str,
    langsmith_tracing: bool,
    langsmith_project: str,
) -> dict[str, Any] | None:
    if str(result.get("status", "")) != "awaiting_review":
        return None
    run_id = str(result.get("run_id", "")).strip()
    if not run_id:
        return None

    st.subheader("Pending Review Action")
    st.caption("This run is paused at a human review gate. You can resolve it directly here.")
    decision_col1, decision_col2 = st.columns(2)
    review_notes = st.text_area(
        "Review Notes",
        height=100,
        key=f"inline_review_notes_{run_id}",
    )
    approve = decision_col1.button("Approve Review", type="primary", key=f"approve_review_{run_id}")
    reject = decision_col2.button("Reject Review", key=f"reject_review_{run_id}")

    if not approve and not reject:
        return None

    decision = "approved" if approve else "rejected"
    try:
        resumed = resume_daily_regime_run(
            run_id=run_id,
            decision=decision,
            notes=review_notes.strip() or None,
            agent_path=agent_path,
            storage_root=storage_root,
            database_url=database_url or None,
            langsmith_tracing=langsmith_tracing,
            langsmith_project=langsmith_project or None,
        )
    except Exception as exc:
        st.error(str(exc))
        return None

    st.success(f"Review was {decision}.")
    return resumed


def _render_ibkr_result(st, result: dict[str, Any]) -> None:
    outputs = dict(result.get("named_outputs", {}))
    snapshot = dict(outputs.get("ibkr_snapshot", {}))
    symbols = dict(snapshot.get("symbols", {}))
    option_chain = dict(snapshot.get("option_chain", {}))
    quotes = list(option_chain.get("option_quotes", []))
    symbol_name = next(iter(symbols), "SPY")
    underlying = dict(symbols.get(symbol_name, {}))

    _render_runtime_diagnostics(st, result)

    st.subheader("Snapshot Summary")
    price_col, exp_col, strike_col, contracts_col = st.columns(4)
    price_col.metric("Underlying", f"{underlying.get('last', 'n/a')}")
    exp_col.metric("Expiries", len(option_chain.get("expirations", [])))
    strike_col.metric("Strikes", len(option_chain.get("strikes", [])))
    contracts_col.metric("Contracts", len(quotes))

    st.subheader("Underlying Quote")
    if symbols:
        st.json(symbols)
    else:
        st.warning("No underlying quote was returned in the workflow outputs.")

    st.subheader("Option Quotes")
    if quotes:
        rows = []
        for quote in quotes:
            greeks = dict(quote.get("greeks", {}))
            rows.append(
                {
                    "symbol": quote.get("symbol"),
                    "expiry": quote.get("expiry"),
                    "strike": quote.get("strike"),
                    "right": quote.get("right"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "last": quote.get("last"),
                    "mark": quote.get("mark"),
                    "volume": quote.get("volume"),
                    "open_interest": quote.get("open_interest"),
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "implied_vol": greeks.get("implied_vol"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No option quotes were returned in this snapshot.")

    with st.expander("Internal Run State"):
        st.code(_pretty_json(result), language="json")


def _replace_last(history: dict[str, list[float]], key: str, value: float) -> None:
    series = list(history.get(key, []))
    if series:
        series[-1] = value
    else:
        series = [value]
    history[key] = series


def _build_scenario_snapshot(
    *,
    base_observation: dict[str, Any],
    vix_value: float,
    vvix_vix_ratio: float,
    term_symbol: str,
    term_value: float,
) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(base_observation))
    symbols = dict(snapshot.get("symbols", {}))
    history = {key: list(value) for key, value in dict(snapshot.get("history", {})).items()}

    symbols.setdefault("VIX", {})["last"] = float(vix_value)
    vvix_value = float(vix_value) * float(vvix_vix_ratio)
    symbols.setdefault("VVIX", {})["last"] = vvix_value
    symbols.setdefault(term_symbol, {})["last"] = float(term_value)

    _replace_last(history, "VIX", float(vix_value))
    _replace_last(history, "VVIX", vvix_value)
    _replace_last(history, term_symbol, float(term_value))

    snapshot["symbols"] = symbols
    snapshot["history"] = history
    snapshot["source"] = "scenario_from_memory"
    quality = dict(snapshot.get("quality", {}))
    warnings = list(quality.get("warnings", []))
    warnings.append("Scenario mode modified VIX-related inputs from the latest live memory snapshot.")
    quality["warnings"] = warnings
    snapshot["quality"] = quality
    provider_metadata = dict(snapshot.get("provider_metadata", {}))
    provider_metadata["scenario_mode"] = True
    provider_metadata["scenario_controls"] = {
        "vix": float(vix_value),
        "vvix_vix_ratio": float(vvix_vix_ratio),
        "term_symbol": term_symbol,
        "term_value": float(term_value),
    }
    snapshot["provider_metadata"] = provider_metadata
    return snapshot


def _load_text_file(path: str) -> str:
    if not path.strip():
        return ""
    file_path = Path(path).resolve()
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def _resolve_existing_backtest_feature_store(config_path: str) -> Path | None:
    try:
        from src.backtest.hmm_replay.replay_config import load_replay_config
        from src.backtest.hmm_replay.replay_dataset import _resolve_feature_store_path
    except Exception:
        return None
    try:
        config = load_replay_config(config_path)
    except Exception:
        return None
    resolved, _ = _resolve_feature_store_path(config.feature_store_path)
    return resolved if resolved.exists() else None


def _load_feature_store_date_bounds(feature_store_path: Path | None) -> tuple[date | None, date | None]:
    if feature_store_path is None or not feature_store_path.exists():
        return (None, None)
    try:
        if feature_store_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(feature_store_path, columns=["date"])
        elif feature_store_path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(feature_store_path, usecols=["date"])
        else:
            return (None, None)
    except Exception:
        return (None, None)
    if frame.empty or "date" not in frame.columns:
        return (None, None)
    parsed = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if parsed.empty:
        return (None, None)
    return (parsed.min().date(), parsed.max().date())


def _load_feature_store_frame(feature_store_path: Path | None) -> pd.DataFrame | None:
    if feature_store_path is None or not feature_store_path.exists():
        return None
    try:
        if feature_store_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(feature_store_path)
        elif feature_store_path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(feature_store_path)
        else:
            return None
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns:
        return frame
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    working = working.dropna(subset=["date"]).reset_index(drop=True)
    return working


def _resolve_historical_cache_db_path() -> Path | None:
    candidates = [
        APP_PATHS.root / "data" / "processed" / "historical_data.db",
        APP_PATHS.root.parent / "data" / "processed" / "historical_data.db",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _load_historical_cache_frame(db_path: Path | None) -> pd.DataFrame | None:
    if db_path is None or not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            frame = pd.read_sql_query(
                """
                SELECT symbol, day, close, source, updated_at
                FROM eod_history
                ORDER BY day DESC, symbol ASC
                """,
                connection,
            )
    except Exception:
        return None
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="coerce").dt.date
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], errors="coerce", utc=True)
    return frame.dropna(subset=["day"]).reset_index(drop=True)


def _clamp_date_to_bounds(value: date, *, minimum: date | None, maximum: date | None) -> date:
    result = value
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


def _load_backtest_config(config_path: str):
    try:
        from src.backtest.hmm_replay.replay_config import load_replay_config
    except Exception:
        return None
    try:
        return load_replay_config(config_path)
    except Exception:
        return None


def _load_policy_backtest_config(config_path: str):
    try:
        from src.backtest.policy.policy_backtester import load_policy_backtest_config
    except Exception:
        return None
    try:
        return load_policy_backtest_config(config_path)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _build_policy_mechanics_checks(
    *,
    policy_daily_path: str,
    policy_trades_path: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    # 1-6: formula sanity checks from spec
    leap_formula = 0.70 * 100.0 * 1.0 * 10.0
    short_call_profit_formula = (1.50 - 0.30) * 100.0
    short_call_loss_formula = (1.50 - 3.00) * 100.0
    combined_formula = leap_formula + short_call_loss_formula
    leap_profit_exit_formula = (120.0 - 100.0) * 100.0
    leap_stop_exit_formula = (80.0 - 100.0) * 100.0
    checks.extend(
        [
            {
                "check": "1) LEAP delta sanity (+10 SPY points -> +$700)",
                "expected": 700.0,
                "actual": round(float(leap_formula), 2),
                "status": "PASS" if abs(leap_formula - 700.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
            {
                "check": "2) Short call profit exit sanity (1.50 -> 0.30 -> +$120)",
                "expected": 120.0,
                "actual": round(float(short_call_profit_formula), 2),
                "status": "PASS" if abs(short_call_profit_formula - 120.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
            {
                "check": "3) Short call loss exit sanity (1.50 -> 3.00 -> -$150)",
                "expected": -150.0,
                "actual": round(float(short_call_loss_formula), 2),
                "status": "PASS" if abs(short_call_loss_formula + 150.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
            {
                "check": "4) Combined sanity (+700 and -150 -> +550)",
                "expected": 550.0,
                "actual": round(float(combined_formula), 2),
                "status": "PASS" if abs(combined_formula - 550.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
            {
                "check": "5) LEAP profit exit sanity (100 -> 120 -> +$2000)",
                "expected": 2000.0,
                "actual": round(float(leap_profit_exit_formula), 2),
                "status": "PASS" if abs(leap_profit_exit_formula - 2000.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
            {
                "check": "6) LEAP stop-loss sanity (100 -> 80 -> -$2000)",
                "expected": -2000.0,
                "actual": round(float(leap_stop_exit_formula), 2),
                "status": "PASS" if abs(leap_stop_exit_formula + 2000.0) < 1e-6 else "FAIL",
                "source": "formula",
            },
        ]
    )

    # 7: run-level no-naked-call invariant and dollar multiplier consistency
    daily_df: pd.DataFrame | None = None
    trades_df: pd.DataFrame | None = None
    try:
        if policy_daily_path and Path(policy_daily_path).exists():
            daily_df = pd.read_csv(policy_daily_path)
    except Exception:
        daily_df = None
    try:
        if policy_trades_path and Path(policy_trades_path).exists():
            trades_df = pd.read_csv(policy_trades_path)
    except Exception:
        trades_df = None

    if daily_df is not None and not daily_df.empty and {"leap_open", "short_call_open"}.issubset(daily_df.columns):
        naked_count = int(((daily_df["leap_open"] == False) & (daily_df["short_call_open"] == True)).sum())  # noqa: E712
        checks.append(
            {
                "check": "7) No naked short calls invariant (LEAP closed => no short call)",
                "expected": 0,
                "actual": naked_count,
                "status": "PASS" if naked_count == 0 else "FAIL",
                "source": "run",
            }
        )
    else:
        checks.append(
            {
                "check": "7) No naked short calls invariant (LEAP closed => no short call)",
                "expected": 0,
                "actual": "n/a",
                "status": "N/A",
                "source": "run",
            }
        )

    if trades_df is not None and not trades_df.empty and {"instrument_type", "entry_premium", "exit_premium", "dollar_pnl"}.issubset(trades_df.columns):
        short_calls = trades_df[trades_df["instrument_type"] == "SHORT_CALL"].copy()
        if not short_calls.empty:
            calc = (pd.to_numeric(short_calls["entry_premium"], errors="coerce") - pd.to_numeric(short_calls["exit_premium"], errors="coerce")) * 100.0
            actual = pd.to_numeric(short_calls["dollar_pnl"], errors="coerce")
            mismatch = int((calc - actual).abs().fillna(0.0).gt(0.02).sum())
            checks.append(
                {
                    "check": "Dollar scaling check (SHORT_CALL dollar_pnl = (entry-exit)*100)",
                    "expected": 0,
                    "actual": mismatch,
                    "status": "PASS" if mismatch == 0 else "FAIL",
                    "source": "run",
                }
            )
    return checks


def main() -> None:
    st = _load_streamlit()

    st.set_page_config(
        page_title="Agentic Vol Regime App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_widget_theme(st)
    st.title("Agentic Vol Regime App")
    st.caption("Decision-support frontend on top of agentic_harness.")

    with st.sidebar:
        st.header("Runtime")
        storage_root = st.text_input(
            "Storage Root",
            value=str((APP_PATHS.root / ".streamlit_workflow_memory").resolve()),
        )
        database_url = st.text_input("Database URL", value="")
        langsmith_tracing = st.checkbox("Enable LangSmith Tracing", value=False)
        langsmith_project = st.text_input("LangSmith Project", value="agentic_harness")
        st.caption("IBKR default port is 4001.")

    daily_tab, historical_tab, backtester_tab, policy_backtester_tab, ibkr_tab, resume_tab = st.tabs(
        [
            "Daily Belief Report",
            "Historical Belief Reports",
            "Backtester",
            "Policy Backtest",
            "IBKR Snapshot Agent",
            "Resume Daily Review",
        ]
    )
    state_daily_result_key = "vol_regime_last_daily_result"
    state_historical_report_key = "vol_regime_historical_report_result"
    state_backtester_result_key = "vol_regime_backtester_result"
    state_backtester_build_key = "vol_regime_backtester_build_result"
    state_policy_backtester_result_key = "vol_regime_policy_backtester_result"

    with daily_tab:
        st.subheader("Daily Volatility Regime Workflow")
        default_daily_payload = load_json(DEFAULT_DAILY_INPUT)
        default_live_payload = load_json(DEFAULT_DAILY_LIVE_INPUT)
        agent_choice = st.selectbox(
            "Regime Engine",
            options=REGIME_ENGINE_OPTIONS,
            help="All agents use the same workflow surface. HMM variants additively extend the HMMv1 core lens.",
        )
        selected_daily_agent_path = _resolve_daily_agent_path_from_choice(agent_choice)
        mode = st.radio(
            "Mode",
            options=["Live IBKR", "Scenario", "Manual JSON"],
            horizontal=True,
            help=(
                "Live IBKR refreshes the regime inputs from IBKR. Scenario mode disables the "
                "live call and perturbs the latest live observation stored in memory."
            ),
        )

        effective_daily_payload: dict[str, Any]
        if mode == "Live IBKR":
            ibkr_defaults = dict(default_live_payload.get("ibkr", {}))
            default_history_days = _default_history_days_for_agent_choice(
                agent_choice,
                int(ibkr_defaults.get("history_days", 252)),
            )
            with st.expander("Advanced IBKR Settings", expanded=False):
                live_col1, live_col2, live_col3 = st.columns(3)
                live_host = live_col1.text_input(
                    "IBKR Host",
                    value=str(ibkr_defaults.get("host", "127.0.0.1")),
                    key="daily_live_host",
                )
                live_port = live_col2.number_input(
                    "IBKR Port",
                    value=int(ibkr_defaults.get("port", 4001)),
                    step=1,
                    key="daily_live_port",
                )
                live_client_id = live_col3.number_input(
                    "Client ID",
                    value=int(ibkr_defaults.get("client_id", 73)),
                    step=1,
                    key="daily_live_client_id",
                )

                live_cfg1, live_cfg2, live_cfg3, live_cfg4 = st.columns(4)
                live_market_data_type = live_cfg1.number_input(
                    "Market Data Type",
                    value=int(ibkr_defaults.get("market_data_type", 1)),
                    step=1,
                    key="daily_live_market_data_type",
                )
                live_history_days = live_cfg2.number_input(
                    "History Days",
                    value=int(default_history_days),
                    step=1,
                    key="daily_live_history_days",
                )
                live_expiry_count = live_cfg3.number_input(
                    "Expiry Count",
                    value=int(ibkr_defaults.get("expiry_count", 2)),
                    step=1,
                    key="daily_live_expiry_count",
                )
                live_strike_count = live_cfg4.number_input(
                    "Strike Count",
                    value=int(ibkr_defaults.get("strike_count", 8)),
                    step=1,
                    key="daily_live_strike_count",
                )

                live_cfg5, live_cfg6, live_cfg7 = st.columns(3)
                live_exchange = live_cfg5.text_input(
                    "Exchange",
                    value=str(ibkr_defaults.get("exchange", "SMART")),
                    key="daily_live_exchange",
                )
                live_option_exchange = live_cfg6.text_input(
                    "Option Exchange",
                    value=str(ibkr_defaults.get("option_exchange", "SMART")),
                    key="daily_live_option_exchange",
                )
                live_index_exchange = live_cfg7.text_input(
                    "Index Exchange",
                    value=str(ibkr_defaults.get("index_exchange", "CBOE")),
                    key="daily_live_index_exchange",
                )

                live_currency = st.text_input(
                    "Currency",
                    value=str(ibkr_defaults.get("currency", "USD")),
                    key="daily_live_currency",
                )

            effective_daily_payload = {
                "data_provider": "ibkr",
                "symbol": str(default_live_payload.get("symbol", "SPY")),
                "ibkr": {
                    "host": live_host.strip() or "127.0.0.1",
                    "port": int(live_port),
                    "client_id": int(live_client_id),
                    "market_data_type": int(live_market_data_type),
                    "exchange": live_exchange.strip() or "SMART",
                    "option_exchange": live_option_exchange.strip() or "SMART",
                    "currency": live_currency.strip() or "USD",
                    "index_exchange": live_index_exchange.strip() or "CBOE",
                    "expiry_count": int(live_expiry_count),
                    "strike_count": int(live_strike_count),
                    "history_days": int(live_history_days),
                    "min_days_to_expiry": 0,
                },
                "reference_market_snapshot": dict(default_daily_payload.get("market_snapshot", {})),
            }
            st.caption("Using default symbol set: SPY, VIX, VVIX, VIX9D, and VIX3M.")
            if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent", "HMMv3.1 Meta-Blend Agent"}:
                st.caption("HMM live runs default to a 756-day history window. Increase this to 1260 if you want a deeper regime-training window.")
        elif mode == "Scenario":
            latest_live_observation = load_latest_live_daily_observation(
                agent_path=selected_daily_agent_path,
                storage_root=storage_root,
                database_url=database_url or None,
            )
            if latest_live_observation is None:
                st.warning(
                    "No live daily observation found in memory yet. Run the Daily Belief Report "
                    "once in Live IBKR mode first."
                )
                effective_daily_payload = dict(default_daily_payload)
            else:
                base_symbols = dict(latest_live_observation.get("symbols", {}))
                base_history = dict(latest_live_observation.get("history", {}))
                term_symbol_options = ["VIX3M", "VIX6M", "VIX9M"]
                base_vix = float(base_symbols.get("VIX", {}).get("last", 15.0))
                base_vvix = float(base_symbols.get("VVIX", {}).get("last", base_vix * 5.5))
                base_ratio = base_vvix / base_vix if base_vix else 5.5
                default_term_symbol = "VIX3M"

                st.caption(
                    f"Scenario base loaded from live memory snapshot at "
                    f"`{latest_live_observation.get('as_of', 'unknown')}`."
                )
                scenario_col1, scenario_col2 = st.columns(2)
                term_symbol = scenario_col1.selectbox(
                    "Term Structure Leg",
                    options=term_symbol_options,
                    index=term_symbol_options.index(default_term_symbol),
                    format_func=lambda value: value.replace("VIX", "VIX ").replace("M", "M"),
                )
                term_base_value = float(
                    base_symbols.get(term_symbol, {}).get(
                        "last",
                        base_symbols.get("VIX3M", {}).get("last", base_vix),
                    )
                )
                vix_value = scenario_col1.slider(
                    "VIX",
                    min_value=8.0,
                    max_value=max(40.0, base_vix + 20.0),
                    value=float(base_vix),
                    step=0.1,
                )
                vvix_vix_ratio = scenario_col2.slider(
                    "VVIX / VIX Ratio",
                    min_value=3.5,
                    max_value=10.0,
                    value=float(base_ratio),
                    step=0.05,
                )
                term_value = scenario_col2.slider(
                    f"{term_symbol}",
                    min_value=8.0,
                    max_value=max(45.0, term_base_value + 20.0),
                    value=float(term_base_value),
                    step=0.1,
                )

                scenario_snapshot = _build_scenario_snapshot(
                    base_observation=latest_live_observation,
                    vix_value=vix_value,
                    vvix_vix_ratio=vvix_vix_ratio,
                    term_symbol=term_symbol,
                    term_value=term_value,
                )
                scenario_snapshot.setdefault("provider_metadata", {})["term_structure_symbol"] = term_symbol
                effective_daily_payload = {
                    "market_snapshot": scenario_snapshot,
                    "report_root": str((APP_PATHS.root / "reports").resolve()),
                }
                with st.expander("Scenario Snapshot"):
                    st.code(_pretty_json(scenario_snapshot), language="json")
        else:
            daily_json = st.text_area(
                "Daily Input JSON",
                value=_pretty_json(default_daily_payload),
                height=360,
                key="daily_json",
            )
            effective_daily_payload = _parse_json_text(daily_json, fallback=default_daily_payload)

        if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent", "HMMv3.1 Meta-Blend Agent"}:
            with st.expander("HMM Snapshot", expanded=False):
                st.caption(
                    "Freeze the current trained HMM artifact and feature-set config before experimenting with new features."
                )
                snapshot_label = st.text_input(
                    "Snapshot Label",
                    value="pre_feature_experiments",
                    key="hmm_snapshot_label",
                )
                if st.button("Create HMM Baseline Snapshot", key="create_hmm_baseline_snapshot", type="secondary"):
                    try:
                        snapshot_result = snapshot_hmm_baseline(
                            snapshot_label=snapshot_label.strip() or None,
                            agent_path=selected_daily_agent_path,
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(
                            "Created HMM snapshot at "
                            f"`{snapshot_result['snapshot_dir']}` "
                            f"with {snapshot_result['feature_count']} features."
                        )
            with st.expander("HMM Reset", expanded=False):
                st.caption(
                    "Clear the HMM agent's cached history, saved HMM state history, latest live observation snapshot, and the persisted HMM model artifact."
                )
                if st.button("Reset HMM Persisted State", key="reset_hmm_persisted_state", type="secondary"):
                    reset_result = reset_hmm_persisted_state(
                        agent_path=selected_daily_agent_path,
                        storage_root=storage_root,
                        database_url=database_url or None,
                    )
                    st.session_state.pop(state_daily_result_key, None)
                    st.success(
                        "Cleared HMM persisted state: "
                        f"{reset_result['deleted_memory_records']} | "
                        f"model_deleted={reset_result['deleted_model_artifact']}"
                    )
                    st.rerun()

        if st.button("Run Daily Workflow", type="primary"):
            try:
                payload = dict(effective_daily_payload)
                result = run_daily_regime_agent(
                    input_payload=payload,
                    agent_path=selected_daily_agent_path,
                    storage_root=storage_root,
                    database_url=database_url or None,
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[state_daily_result_key] = result

        current_daily_result = st.session_state.get(state_daily_result_key)
        if isinstance(current_daily_result, dict):
            resumed_result = _render_inline_review_actions(
                st,
                result=current_daily_result,
                agent_path=Path(selected_daily_agent_path),
                storage_root=storage_root,
                database_url=database_url,
                langsmith_tracing=langsmith_tracing,
                langsmith_project=langsmith_project,
            )
            if isinstance(resumed_result, dict):
                st.session_state[state_daily_result_key] = resumed_result
                current_daily_result = resumed_result
            hmm_history = (
                load_recent_hmm_state_history(
                    agent_path=selected_daily_agent_path,
                    storage_root=storage_root,
                    database_url=database_url or None,
                    limit=12,
                )
                if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent", "HMMv3.1 Meta-Blend Agent"}
                else []
            )
            _render_daily_result(st, current_daily_result, hmm_history=hmm_history)

    with historical_tab:
        st.subheader("Historical Belief Reports")
        st.caption(
            "Load a previously generated report for a specific date and model. "
            "If it does not exist yet, the app will run the selected model using that as-of date."
        )

        historical_default_live_payload = load_json(DEFAULT_DAILY_LIVE_INPUT)
        historical_agent_choice = st.selectbox(
            "Historical Regime Engine",
            options=REGIME_ENGINE_OPTIONS,
            key="historical_regime_engine",
        )
        selected_historical_agent_path = _resolve_daily_agent_path_from_choice(historical_agent_choice)
        historical_date = st.date_input(
            "Historical As-of Date",
            value=date.today(),
            max_value=date.today(),
            key="historical_as_of_date",
        )

        historical_ibkr_defaults = dict(historical_default_live_payload.get("ibkr", {}))
        historical_default_history_days = _default_history_days_for_agent_choice(
            historical_agent_choice,
            int(historical_ibkr_defaults.get("history_days", 252)),
        )
        with st.expander("Advanced Historical IBKR Settings", expanded=False):
            hist_col1, hist_col2, hist_col3 = st.columns(3)
            historical_host = hist_col1.text_input(
                "IBKR Host",
                value=str(historical_ibkr_defaults.get("host", "127.0.0.1")),
                key="historical_live_host",
            )
            historical_port = hist_col2.number_input(
                "IBKR Port",
                value=int(historical_ibkr_defaults.get("port", 4001)),
                step=1,
                key="historical_live_port",
            )
            historical_client_id = hist_col3.number_input(
                "Client ID",
                value=int(historical_ibkr_defaults.get("client_id", 73)),
                step=1,
                key="historical_live_client_id",
            )

            hist_cfg1, hist_cfg2, hist_cfg3, hist_cfg4 = st.columns(4)
            historical_market_data_type = hist_cfg1.number_input(
                "Market Data Type",
                value=int(historical_ibkr_defaults.get("market_data_type", 1)),
                step=1,
                key="historical_market_data_type",
            )
            historical_history_days = hist_cfg2.number_input(
                "History Days",
                value=int(historical_default_history_days),
                step=1,
                key="historical_history_days",
            )
            historical_expiry_count = hist_cfg3.number_input(
                "Expiry Count",
                value=int(historical_ibkr_defaults.get("expiry_count", 2)),
                step=1,
                key="historical_expiry_count",
            )
            historical_strike_count = hist_cfg4.number_input(
                "Strike Count",
                value=int(historical_ibkr_defaults.get("strike_count", 8)),
                step=1,
                key="historical_strike_count",
            )

            hist_cfg5, hist_cfg6, hist_cfg7 = st.columns(3)
            historical_exchange = hist_cfg5.text_input(
                "Exchange",
                value=str(historical_ibkr_defaults.get("exchange", "SMART")),
                key="historical_exchange",
            )
            historical_option_exchange = hist_cfg6.text_input(
                "Option Exchange",
                value=str(historical_ibkr_defaults.get("option_exchange", "SMART")),
                key="historical_option_exchange",
            )
            historical_index_exchange = hist_cfg7.text_input(
                "Index Exchange",
                value=str(historical_ibkr_defaults.get("index_exchange", "CBOE")),
                key="historical_index_exchange",
            )

            historical_currency = st.text_input(
                "Currency",
                value=str(historical_ibkr_defaults.get("currency", "USD")),
                key="historical_currency",
            )

        if st.button("Load Historical Report", type="primary", key="load_historical_report_button"):
            historical_payload = {
                "data_provider": "ibkr",
                "symbol": str(historical_default_live_payload.get("symbol", "SPY")),
                "ibkr": {
                    "host": historical_host.strip() or "127.0.0.1",
                    "port": int(historical_port),
                    "client_id": int(historical_client_id),
                    "market_data_type": int(historical_market_data_type),
                    "exchange": historical_exchange.strip() or "SMART",
                    "option_exchange": historical_option_exchange.strip() or "SMART",
                    "currency": historical_currency.strip() or "USD",
                    "index_exchange": historical_index_exchange.strip() or "CBOE",
                    "expiry_count": int(historical_expiry_count),
                    "strike_count": int(historical_strike_count),
                    "history_days": int(historical_history_days),
                    "min_days_to_expiry": 0,
                },
                "reference_market_snapshot": dict(default_daily_payload.get("market_snapshot", {})),
            }
            try:
                historical_result = load_or_run_historical_belief_report(
                    as_of_date=historical_date.isoformat(),
                    input_payload=historical_payload,
                    agent_path=selected_historical_agent_path,
                    report_root=APP_PATHS.reports_dir,
                    storage_root=storage_root,
                    database_url=database_url or None,
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[state_historical_report_key] = historical_result

        current_historical_result = st.session_state.get(state_historical_report_key)
        if isinstance(current_historical_result, dict):
            historical_source = str(current_historical_result.get("source", "unknown"))
            historical_report_path = str(current_historical_result.get("report_path") or "").strip()
            historical_markdown = str(current_historical_result.get("markdown") or "").strip()
            historical_run_result = current_historical_result.get("run_result")

            if historical_source == "history":
                st.success("Loaded report from history.")
            elif historical_source == "run":
                st.success("Generated report because no historical report existed for that date/model.")
            elif historical_source == "legacy_markdown_history":
                st.warning(
                    "Found only a legacy markdown artifact for this date/model. "
                    "Run the historical report again to backfill the full structured artifact into harness memory."
                )

            if historical_report_path:
                st.caption(f"Report path: `{historical_report_path}`")

            if isinstance(historical_run_result, dict) and historical_run_result:
                _render_daily_result(st, historical_run_result, hmm_history=[])
            elif historical_markdown:
                st.subheader("Historical Report")
                st.markdown(_strip_report_heading_and_summary(historical_markdown))

    with backtester_tab:
        st.subheader("Backtester")
        st.caption(
            "Deterministic historical replay with strict no-lookahead constraints. "
            "This mode is fully offline and does not call IBKR."
        )

        config_path = st.text_input(
            "Replay Config Path",
            value=str(DEFAULT_BACKTEST_CONFIG.resolve()),
            key="backtest_config_path",
        )
        replay_config = _load_backtest_config(config_path.strip())
        strict_10y_mode = bool(getattr(replay_config, "require_10y_replay", False))
        train_lookback_days = int(getattr(replay_config, "train_lookback_days", 2520))
        min_history_days = max(2520 if strict_10y_mode else 252, train_lookback_days)
        default_history_days = max(2520, min_history_days) if strict_10y_mode else max(2520, train_lookback_days)

        if strict_10y_mode:
            st.info(
                "Strict 10-year replay is enabled. Replay runs offline from the local feature store. "
                "To satisfy `start_date=2016-01-01` with `train_lookback_days=2520`, the feature store must "
                "typically reach back to around 2013-01-01 or earlier. "
                "Use `History Days` around 3300-3600 for IBKR builds."
            )
        st.markdown("**Feature Store Builder (IBKR)**")
        default_live_payload = load_json(DEFAULT_DAILY_LIVE_INPUT)
        builder_ibkr = dict(default_live_payload.get("ibkr", {}))
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        backtest_symbol = bcol1.text_input("Symbol", value=str(default_live_payload.get("symbol", "SPY")), key="backtest_build_symbol")
        backtest_history_days = bcol2.number_input(
            "History Days (IBKR Pull)",
            value=default_history_days,
            min_value=min_history_days,
            step=1,
            key="backtest_build_history_days",
            help=(
                "This controls how much historical data is requested from IBKR to build the local feature store. "
                "In strict 10-year mode, this must be large enough to include pre-2016 training history."
            ),
        )
        backtest_host = bcol3.text_input("IBKR Host", value=str(builder_ibkr.get("host", "127.0.0.1")), key="backtest_build_host")
        backtest_port = bcol4.number_input("IBKR Port", value=int(builder_ibkr.get("port", 4001)), step=1, key="backtest_build_port")
        bcol5, bcol6, bcol7, bcol8 = st.columns(4)
        backtest_client_id = bcol5.number_input("Client ID", value=int(builder_ibkr.get("client_id", 73)), step=1, key="backtest_build_client_id")
        backtest_market_data_type = bcol6.number_input("Market Data Type", value=int(builder_ibkr.get("market_data_type", 1)), step=1, key="backtest_build_market_data_type")
        backtest_exchange = bcol7.text_input("Exchange", value=str(builder_ibkr.get("exchange", "SMART")), key="backtest_build_exchange")
        backtest_index_exchange = bcol8.text_input("Index Exchange", value=str(builder_ibkr.get("index_exchange", "CBOE")), key="backtest_build_index_exchange")
        backtest_as_of = st.date_input(
            "Feature Store As-of Date (optional historical cut)",
            value=date.today(),
            max_value=date.today(),
            key="backtest_build_as_of",
            help="The builder fetches up to this date from IBKR. Use today's date for current replay data.",
        )
        if st.button("Build / Refresh Feature Store", type="secondary", key="build_backtest_feature_store_button"):
            try:
                build_result = build_backtest_feature_store(
                    config_path=config_path.strip(),
                    history_days=int(backtest_history_days),
                    as_of_date=backtest_as_of.isoformat(),
                    symbol=backtest_symbol.strip() or "SPY",
                    host=backtest_host.strip() or "127.0.0.1",
                    port=int(backtest_port),
                    client_id=int(backtest_client_id),
                    market_data_type=int(backtest_market_data_type),
                    exchange=backtest_exchange.strip() or "SMART",
                    option_exchange=str(builder_ibkr.get("option_exchange", "SMART")),
                    index_exchange=backtest_index_exchange.strip() or "CBOE",
                    currency=str(builder_ibkr.get("currency", "USD")),
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[state_backtester_build_key] = build_result

        current_build_result = st.session_state.get(state_backtester_build_key)
        if isinstance(current_build_result, dict):
            st.caption(
                "Feature store ready: "
                f"`{current_build_result.get('feature_store_path', '')}` "
                f"| rows={current_build_result.get('rows', 0)} "
                f"| {current_build_result.get('start_date', '')} -> {current_build_result.get('end_date', '')}"
            )
            coverage_report_path = str(current_build_result.get("coverage_report_path", "") or "")
            if coverage_report_path:
                st.caption(f"Coverage report: `{coverage_report_path}`")
            build_warnings = list(current_build_result.get("warnings", []))
            if build_warnings:
                st.warning(" | ".join(str(item) for item in build_warnings))
            required_summary = dict(current_build_result.get("required_history_summary", {}))
            if required_summary:
                st.caption(
                    "Required-series coverage: "
                    f"min_rows={required_summary.get('min_required_rows', 0)} "
                    f"| inferred window {required_summary.get('inferred_required_start_date', '')} -> "
                    f"{required_summary.get('inferred_required_end_date', '')}"
                )
                truncating_keys = list(required_summary.get("truncating_required_keys", []))
                if truncating_keys:
                    st.warning(
                        "Truncating required series: "
                        + ", ".join(str(item) for item in truncating_keys)
                    )
            history_coverage = list(current_build_result.get("history_coverage", []))
            if history_coverage:
                with st.expander("IBKR History Coverage Diagnostics", expanded=False):
                    st.dataframe(pd.DataFrame(history_coverage), width="stretch", hide_index=True)
            source_quality = dict(current_build_result.get("source_quality", {}))
            if source_quality:
                with st.expander("IBKR Source Quality Payload", expanded=False):
                    st.json(source_quality)

        resolved_store = _resolve_existing_backtest_feature_store(config_path.strip())
        if resolved_store is not None:
            st.caption(f"Feature store: `{resolved_store}`")
        else:
            st.warning(
                "Feature store file is missing for the current config. "
                "Update `feature_store_path` in the replay config to a valid parquet/csv dataset."
            )
        feature_min_date, feature_max_date = _load_feature_store_date_bounds(resolved_store)
        if feature_min_date is not None and feature_max_date is not None:
            st.caption(f"Feature-store window: `{feature_min_date.isoformat()}` to `{feature_max_date.isoformat()}`")
            st.markdown("**Historical Data Visualizer**")
            feature_frame = _load_feature_store_frame(resolved_store)
            if feature_frame is None or feature_frame.empty:
                st.warning("Feature-store data could not be loaded for table visualization.")
            else:
                default_start = feature_min_date
                default_end = feature_max_date
                viz_col1, viz_col2 = st.columns(2)
                selected_start = viz_col1.date_input(
                    "Table Start Date",
                    value=default_start,
                    min_value=feature_min_date,
                    max_value=feature_max_date,
                    key="backtest_table_start_date",
                )
                selected_end = viz_col2.date_input(
                    "Table End Date",
                    value=default_end,
                    min_value=feature_min_date,
                    max_value=feature_max_date,
                    key="backtest_table_end_date",
                )
                selected_start = _clamp_date_to_bounds(selected_start, minimum=feature_min_date, maximum=feature_max_date)
                selected_end = _clamp_date_to_bounds(selected_end, minimum=feature_min_date, maximum=feature_max_date)
                if selected_start > selected_end:
                    selected_start = selected_end

                control_col1, control_col2 = st.columns(2)
                max_rows = control_col1.number_input(
                    "Max Rows to Display",
                    min_value=50,
                    max_value=10000,
                    value=500,
                    step=50,
                    key="backtest_table_max_rows",
                )
                sort_order = control_col2.selectbox(
                    "Sort by Date",
                    options=["Descending", "Ascending"],
                    index=0,
                    key="backtest_table_sort_order",
                )

                available_columns = list(feature_frame.columns)
                default_columns = [
                    column
                    for column in [
                        "date",
                        "spy_close",
                        "vix",
                        "vvix",
                        "vix3m",
                        "realized_vol_21d",
                        "vvix_vix_ratio",
                        "term_structure_slope",
                        "regime_target",
                    ]
                    if column in available_columns
                ]
                if not default_columns:
                    default_columns = available_columns[: min(12, len(available_columns))]
                selected_columns = st.multiselect(
                    "Columns",
                    options=available_columns,
                    default=default_columns,
                    key="backtest_table_columns",
                )
                if not selected_columns:
                    selected_columns = default_columns

                scoped_frame = feature_frame[
                    (feature_frame["date"] >= selected_start) & (feature_frame["date"] <= selected_end)
                ].copy()
                scoped_frame = scoped_frame.sort_values(
                    "date",
                    ascending=(sort_order == "Ascending"),
                ).reset_index(drop=True)
                st.caption(
                    f"Rows in selected window: `{len(scoped_frame)}` | "
                    f"displaying up to `{int(max_rows)}` rows."
                )
                st.dataframe(
                    scoped_frame[selected_columns].head(int(max_rows)),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown("**Historical EOD Cache (SQLite)**")
                cache_db_path = _resolve_historical_cache_db_path()
                if cache_db_path is None:
                    st.warning("Historical cache DB not found (`historical_data.db`).")
                else:
                    st.caption(f"Cache DB: `{cache_db_path}`")
                    cache_frame = _load_historical_cache_frame(cache_db_path)
                    if cache_frame is None:
                        st.warning("Could not load the historical cache table from SQLite.")
                    elif cache_frame.empty:
                        st.info("Historical cache exists but contains no rows.")
                    else:
                        cache_min_date = min(cache_frame["day"])
                        cache_max_date = max(cache_frame["day"])
                        cache_col1, cache_col2 = st.columns(2)
                        cache_start = cache_col1.date_input(
                            "Cache Start Date",
                            value=cache_min_date,
                            min_value=cache_min_date,
                            max_value=cache_max_date,
                            key="historical_cache_start_date",
                        )
                        cache_end = cache_col2.date_input(
                            "Cache End Date",
                            value=cache_max_date,
                            min_value=cache_min_date,
                            max_value=cache_max_date,
                            key="historical_cache_end_date",
                        )
                        if cache_start > cache_end:
                            cache_start = cache_end

                        symbol_options = sorted(str(item) for item in cache_frame["symbol"].dropna().unique().tolist())
                        source_options = sorted(str(item) for item in cache_frame["source"].dropna().unique().tolist())
                        filter_col1, filter_col2 = st.columns(2)
                        selected_symbols = filter_col1.multiselect(
                            "Symbols",
                            options=symbol_options,
                            default=symbol_options[: min(12, len(symbol_options))] if symbol_options else [],
                            key="historical_cache_symbols",
                        )
                        selected_sources = filter_col2.multiselect(
                            "Sources",
                            options=source_options,
                            default=source_options,
                            key="historical_cache_sources",
                        )
                        row_col1, row_col2 = st.columns(2)
                        cache_max_rows = row_col1.number_input(
                            "Cache Max Rows",
                            min_value=50,
                            max_value=20000,
                            value=1000,
                            step=50,
                            key="historical_cache_max_rows",
                        )
                        cache_sort = row_col2.selectbox(
                            "Cache Sort",
                            options=["Newest First", "Oldest First"],
                            index=0,
                            key="historical_cache_sort",
                        )

                        filtered_cache = cache_frame[
                            (cache_frame["day"] >= cache_start) & (cache_frame["day"] <= cache_end)
                        ].copy()
                        if selected_symbols:
                            filtered_cache = filtered_cache[filtered_cache["symbol"].isin(selected_symbols)]
                        if selected_sources:
                            filtered_cache = filtered_cache[filtered_cache["source"].isin(selected_sources)]
                        filtered_cache = filtered_cache.sort_values(
                            ["day", "symbol"],
                            ascending=(cache_sort == "Oldest First"),
                        ).reset_index(drop=True)
                        st.caption(
                            f"Cache rows in selected window/filter: `{len(filtered_cache)}` | "
                            f"displaying up to `{int(cache_max_rows)}` rows."
                        )
                        st.dataframe(
                            filtered_cache[["symbol", "day", "close", "source", "updated_at"]].head(int(cache_max_rows)),
                            width="stretch",
                            hide_index=True,
                        )
        default_backtest_day = feature_max_date or date.today()
        selected_replay_run_mode = st.selectbox(
            "Run Mode",
            options=["tuning", "testing"],
            index=0,
            key="backtest_run_mode",
            help="Use tuning for fast iteration and testing for full validation.",
        )

        if st.session_state.get("backtest_last_run_mode") != selected_replay_run_mode:
            st.session_state["backtest_models"] = _default_backtest_models_for_run_mode(selected_replay_run_mode)
            st.session_state["backtest_horizons"] = _default_backtest_horizons_for_run_mode(selected_replay_run_mode)
            st.session_state["backtest_lightweight_mode"] = selected_replay_run_mode == "tuning"
            st.session_state["backtest_last_run_mode"] = selected_replay_run_mode

        replay_scope = st.radio(
            "Replay Scope",
            options=["Date Range", "Single Date"],
            horizontal=True,
            key="backtest_mode",
        )

        if "backtest_models" not in st.session_state:
            st.session_state["backtest_models"] = _default_backtest_models_for_run_mode(selected_replay_run_mode)
        if "backtest_horizons" not in st.session_state:
            st.session_state["backtest_horizons"] = _default_backtest_horizons_for_run_mode(selected_replay_run_mode)
        if "backtest_lightweight_mode" not in st.session_state:
            st.session_state["backtest_lightweight_mode"] = selected_replay_run_mode == "tuning"

        backtest_models = st.multiselect(
            "Models",
            options=BACKTEST_MODEL_OPTIONS,
            key="backtest_models",
        )
        backtest_horizons = st.multiselect(
            "Horizons (trading days)",
            options=BACKTEST_HORIZON_OPTIONS,
            key="backtest_horizons",
        )
        lightweight_mode = st.checkbox(
            "Lightweight Post-Processing (Recommended)",
            key="backtest_lightweight_mode",
            help=(
                "Skips heavy disagreement/geometry report sections to return results faster and reduce "
                "end-of-run UI stalls. Core replay scoring remains unchanged."
            ),
        )
        preset_rows = _build_run_mode_preset_rows(
            selected_replay_run_mode=selected_replay_run_mode,
            replay_scope=replay_scope,
            selected_models=list(backtest_models),
            selected_horizons=[int(item) for item in list(backtest_horizons)],
            config_train_lookback_days=int(getattr(replay_config, "train_lookback_days", 756) if replay_config else 756),
            config_min_train_rows=int(getattr(replay_config, "min_train_rows", 504) if replay_config else 504),
        )
        with st.expander("Run Mode Preset Summary", expanded=False):
            st.dataframe(preset_rows, width="stretch", hide_index=True)

        as_of_date: str | None = None
        start_date: str | None = None
        end_date: str | None = None
        if replay_scope == "Single Date":
            selected_as_of = st.date_input(
                "As-of Date",
                value=default_backtest_day,
                min_value=feature_min_date,
                max_value=feature_max_date or date.today(),
                key="backtest_as_of_date",
            )
            clamped_as_of = _clamp_date_to_bounds(
                selected_as_of,
                minimum=feature_min_date,
                maximum=feature_max_date,
            )
            if clamped_as_of != selected_as_of:
                st.warning(
                    "Selected as-of date is outside feature-store coverage. "
                    f"Using `{clamped_as_of.isoformat()}`."
                )
            as_of_date = clamped_as_of.isoformat()
        else:
            default_start = feature_min_date or default_backtest_day
            date_col1, date_col2 = st.columns(2)
            selected_start = date_col1.date_input(
                "Start Date",
                value=default_start,
                min_value=feature_min_date,
                max_value=feature_max_date or date.today(),
                key="backtest_start_date",
            )
            selected_end = date_col2.date_input(
                "End Date",
                value=default_backtest_day,
                min_value=feature_min_date,
                max_value=feature_max_date or date.today(),
                key="backtest_end_date",
            )
            clamped_start = _clamp_date_to_bounds(
                selected_start,
                minimum=feature_min_date,
                maximum=feature_max_date,
            )
            clamped_end = _clamp_date_to_bounds(
                selected_end,
                minimum=feature_min_date,
                maximum=feature_max_date,
            )
            if clamped_start > clamped_end:
                clamped_start = clamped_end
                st.warning(
                    "Start date was after end date after clamping to feature-store coverage. "
                    f"Using `{clamped_start.isoformat()}` to `{clamped_end.isoformat()}`."
                )
            elif clamped_start != selected_start or clamped_end != selected_end:
                st.warning(
                    "Date range was clamped to feature-store coverage. "
                    f"Using `{clamped_start.isoformat()}` to `{clamped_end.isoformat()}`."
                )
            start_date = clamped_start.isoformat()
            end_date = clamped_end.isoformat()

        if st.button("Run Backtest", type="primary", key="run_backtest_button"):
            try:
                backtest_result = run_hmm_replay_backtester(
                    config_path=config_path.strip(),
                    run_mode=selected_replay_run_mode,
                    start_date=start_date,
                    end_date=end_date,
                    as_of_date=as_of_date,
                    models=list(backtest_models) or None,
                    horizons=[int(item) for item in list(backtest_horizons)] or None,
                    lightweight_mode=bool(lightweight_mode),
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[state_backtester_result_key] = _compact_backtest_session_payload(backtest_result)

        current_backtest_result = st.session_state.get(state_backtester_result_key)
        if isinstance(current_backtest_result, dict):
            summary_metrics = list(current_backtest_result.get("summary_metrics", []))
            current_mode = str(current_backtest_result.get("run_mode", selected_replay_run_mode)).strip().lower() or "testing"
            display_limit = 50 if current_mode == "tuning" else 100
            st.caption(f"Run Mode: `{current_mode}`")
            st.caption(f"Report: `{current_backtest_result.get('report_path', '')}`")
            st.caption(f"Compact Summary: `{current_backtest_result.get('compact_summary_path', '')}`")
            if current_backtest_result.get("run_log_path"):
                st.caption(f"Run Log: `{current_backtest_result.get('run_log_path', '')}`")
            peak_rss_mb = current_backtest_result.get("peak_rss_mb")
            if peak_rss_mb is not None:
                st.caption(f"Peak RSS Memory: `{peak_rss_mb} MB`")
            runtime_profile = dict(current_backtest_result.get("runtime_profile", {}))
            if runtime_profile:
                st.subheader("Replay Runtime Profile")
                runtime_rows = [
                    {"stage": key, "seconds": runtime_profile.get(key)}
                    for key in [
                        "feature_load_seconds",
                        "feature_engineering_seconds",
                        "model_fit_seconds",
                        "prediction_seconds",
                        "scoring_seconds",
                        "report_generation_seconds",
                        "total_seconds",
                    ]
                    if key in runtime_profile
                ]
                if runtime_rows:
                    st.dataframe(runtime_rows, width="stretch", hide_index=True)
            slowest_dates = list(current_backtest_result.get("slowest_dates", []))
            if slowest_dates:
                st.caption(
                    "Slowest replay dates: "
                    + " | ".join(
                        f"{str(item.get('as_of_date', ''))}: {float(item.get('elapsed_seconds', 0.0)):.2f}s"
                        for item in slowest_dates[:5]
                    )
                )
            st.caption(f"Summary CSV: `{current_backtest_result.get('summary_metrics_path', '')}`")
            st.caption(f"Economic Summary CSV: `{current_backtest_result.get('economic_summary_path', '')}`")
            st.caption(f"Prediction Distribution CSV: `{current_backtest_result.get('prediction_distribution_path', '')}`")
            st.caption(f"Outcome Distribution CSV: `{current_backtest_result.get('outcome_distribution_path', '')}`")
            st.caption(f"Confusion Matrix CSV: `{current_backtest_result.get('confusion_matrix_path', '')}`")
            st.caption(f"False Alarms CSV: `{current_backtest_result.get('false_alarms_path', '')}`")
            st.caption(f"Missed Risks CSV: `{current_backtest_result.get('missed_risks_path', '')}`")
            st.caption(f"Top Feature Importances CSV: `{current_backtest_result.get('top_feature_importances_path', '')}`")
            st.caption(f"Predictions JSONL: `{current_backtest_result.get('prediction_records_path', '')}`")
            st.caption(f"Outcomes JSONL: `{current_backtest_result.get('outcome_records_path', '')}`")
            st.caption(f"Scored JSONL: `{current_backtest_result.get('scored_records_path', '')}`")

            if summary_metrics:
                st.subheader("Summary Metrics")
                st.caption(f"Displaying up to `{display_limit}` rows.")
                st.dataframe(summary_metrics[:display_limit], width="stretch", hide_index=True)

    with policy_backtester_tab:
        st.subheader("Policy Backtester")
        st.caption(
            "V1 economic approximation: option prices are model-estimated (Black-Scholes proxy), "
            "not exact historical option-chain fills."
        )
        policy_config_path = st.text_input(
            "Policy Config Path",
            value=str(DEFAULT_POLICY_BACKTEST_CONFIG.resolve()),
            key="policy_backtest_config_path",
        )
        policy_config = _load_policy_backtest_config(policy_config_path.strip())
        if policy_config is None:
            st.warning("Could not load policy backtest config. Using built-in defaults for this UI run.")

        config_feature_store = (
            str(getattr(policy_config, "feature_store_path", "agentic_vol_regime_app/data/processed/features_daily.parquet"))
        )
        config_run_mode = str(getattr(policy_config, "run_mode", "tuning")).strip().lower()
        config_models = list(getattr(policy_config, "models", POLICY_MODEL_OPTIONS) or POLICY_MODEL_OPTIONS)
        config_start_date = pd.to_datetime(str(getattr(policy_config, "start_date", "2024-01-01")), errors="coerce")
        config_end_text = str(getattr(policy_config, "end_date", "latest")).strip().lower()
        config_end_date = (
            pd.Timestamp.today()
            if config_end_text == "latest"
            else pd.to_datetime(config_end_text, errors="coerce")
        )
        if pd.isna(config_start_date):
            config_start_date = pd.Timestamp("2024-01-01")
        if pd.isna(config_end_date):
            config_end_date = pd.Timestamp.today()

        if st.button("Reset to Config Defaults", key="policy_backtest_reset_defaults_button"):
            st.session_state["policy_backtest_feature_store_path"] = str(
                Path(config_feature_store).resolve()
                if not Path(config_feature_store).is_absolute()
                else Path(config_feature_store)
            )
            st.session_state["policy_backtest_run_mode"] = "tuning" if config_run_mode == "tuning" else "testing"
            st.session_state["policy_backtest_models"] = [
                item for item in config_models if item in POLICY_MODEL_OPTIONS
            ] or list(POLICY_MODEL_OPTIONS)
            st.session_state["policy_backtest_start_date"] = config_start_date.date()
            st.session_state["policy_backtest_end_date"] = config_end_date.date()
            st.session_state["policy_backtest_train_lookback"] = int(getattr(policy_config, "train_lookback_days", 756))
            st.session_state["policy_backtest_min_train_rows"] = int(getattr(policy_config, "min_train_rows", 504))
            st.session_state["policy_backtest_default_dte"] = int(getattr(policy_config, "default_dte", 1))
            st.session_state["policy_backtest_leap_delta"] = float(getattr(policy_config, "leap_delta", 0.75))
            st.session_state["policy_backtest_profit_exit_pct"] = float(getattr(policy_config, "profit_exit_pct", 0.20))
            st.session_state["policy_backtest_loss_exit_multiple"] = float(
                getattr(policy_config, "loss_exit_multiple", 2.0)
            )
            st.session_state["policy_backtest_touch_exit"] = bool(
                getattr(policy_config, "exit_on_underlying_touch", True)
            )

        if "policy_backtest_feature_store_path" not in st.session_state:
            st.session_state["policy_backtest_feature_store_path"] = str(
                Path(config_feature_store).resolve()
                if not Path(config_feature_store).is_absolute()
                else Path(config_feature_store)
            )
        if "policy_backtest_run_mode" not in st.session_state:
            st.session_state["policy_backtest_run_mode"] = "tuning" if config_run_mode == "tuning" else "testing"
        if "policy_backtest_models" not in st.session_state:
            st.session_state["policy_backtest_models"] = [
                item for item in config_models if item in POLICY_MODEL_OPTIONS
            ] or list(POLICY_MODEL_OPTIONS)
        if "policy_backtest_start_date" not in st.session_state:
            st.session_state["policy_backtest_start_date"] = config_start_date.date()
        if "policy_backtest_end_date" not in st.session_state:
            st.session_state["policy_backtest_end_date"] = config_end_date.date()
        if "policy_backtest_train_lookback" not in st.session_state:
            st.session_state["policy_backtest_train_lookback"] = int(getattr(policy_config, "train_lookback_days", 756))
        if "policy_backtest_min_train_rows" not in st.session_state:
            st.session_state["policy_backtest_min_train_rows"] = int(getattr(policy_config, "min_train_rows", 504))
        if "policy_backtest_default_dte" not in st.session_state:
            st.session_state["policy_backtest_default_dte"] = int(getattr(policy_config, "default_dte", 1))
        if "policy_backtest_leap_delta" not in st.session_state:
            st.session_state["policy_backtest_leap_delta"] = float(getattr(policy_config, "leap_delta", 0.75))
        if "policy_backtest_profit_exit_pct" not in st.session_state:
            st.session_state["policy_backtest_profit_exit_pct"] = float(getattr(policy_config, "profit_exit_pct", 0.20))
        if "policy_backtest_loss_exit_multiple" not in st.session_state:
            st.session_state["policy_backtest_loss_exit_multiple"] = float(
                getattr(policy_config, "loss_exit_multiple", 2.0)
            )
        if "policy_backtest_touch_exit" not in st.session_state:
            st.session_state["policy_backtest_touch_exit"] = bool(
                getattr(policy_config, "exit_on_underlying_touch", True)
            )

        policy_col1, policy_col2 = st.columns(2)
        policy_feature_store_path = policy_col1.text_input(
            "Feature Store Path",
            key="policy_backtest_feature_store_path",
        )
        policy_run_mode = policy_col2.selectbox(
            "Run Mode",
            options=["tuning", "testing"],
            key="policy_backtest_run_mode",
        )
        policy_models = st.multiselect(
            "Model Policies (baselines are always included)",
            options=POLICY_MODEL_OPTIONS,
            key="policy_backtest_models",
        )
        policy_dcol1, policy_dcol2 = st.columns(2)
        policy_start_date = policy_dcol1.date_input(
            "Start Date",
            key="policy_backtest_start_date",
        )
        policy_end_date = policy_dcol2.date_input(
            "End Date",
            key="policy_backtest_end_date",
        )
        policy_rcol1, policy_rcol2, policy_rcol3, policy_rcol4 = st.columns(4)
        policy_lookback = policy_rcol1.number_input(
            "Train Lookback",
            min_value=252,
            max_value=4000,
            step=21,
            key="policy_backtest_train_lookback",
        )
        policy_min_rows = policy_rcol2.number_input(
            "Min Train Rows",
            min_value=126,
            max_value=3000,
            step=21,
            key="policy_backtest_min_train_rows",
        )
        policy_dte = policy_rcol3.number_input(
            "Default DTE",
            min_value=1,
            max_value=10,
            step=1,
            key="policy_backtest_default_dte",
        )
        policy_leap_delta = policy_rcol4.number_input(
            "LEAP Delta",
            min_value=0.1,
            max_value=1.0,
            step=0.05,
            key="policy_backtest_leap_delta",
        )
        policy_ecol1, policy_ecol2, policy_ecol3 = st.columns(3)
        policy_profit_exit = policy_ecol1.number_input(
            "Profit Exit Pct",
            min_value=0.05,
            max_value=0.95,
            step=0.05,
            key="policy_backtest_profit_exit_pct",
        )
        policy_loss_multiple = policy_ecol2.number_input(
            "Loss Exit Multiple",
            min_value=1.0,
            max_value=5.0,
            step=0.25,
            key="policy_backtest_loss_exit_multiple",
        )
        policy_touch_exit = policy_ecol3.checkbox(
            "Exit On Underlying Touch",
            key="policy_backtest_touch_exit",
        )
        if st.button("Run Policy Backtest", type="primary", key="run_policy_backtest_button"):
            try:
                policy_result = run_policy_backtester(
                    config_path=policy_config_path.strip() or None,
                    feature_store_path=policy_feature_store_path.strip(),
                    run_mode=policy_run_mode,
                    start_date=policy_start_date.isoformat(),
                    end_date=policy_end_date.isoformat(),
                    models=list(policy_models) or None,
                    train_lookback_days=int(policy_lookback),
                    min_train_rows=int(policy_min_rows),
                    default_dte=int(policy_dte),
                    leap_delta=float(policy_leap_delta),
                    profit_exit_pct=float(policy_profit_exit),
                    loss_exit_multiple=float(policy_loss_multiple),
                    exit_on_underlying_touch=bool(policy_touch_exit),
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state[state_policy_backtester_result_key] = dict(policy_result)

        current_policy_result = st.session_state.get(state_policy_backtester_result_key)
        if isinstance(current_policy_result, dict):
            st.caption(f"Run Mode: `{current_policy_result.get('run_mode', '')}`")
            st.caption(f"Date Window: `{current_policy_result.get('date_start', '')}` -> `{current_policy_result.get('date_end', '')}`")
            st.caption(f"Output Directory: `{current_policy_result.get('output_dir', '')}`")
            st.caption(f"Report: `{current_policy_result.get('report_path', '')}`")
            st.caption(f"Trades CSV: `{current_policy_result.get('policy_trades_path', '')}`")
            st.caption(f"Daily PnL CSV: `{current_policy_result.get('policy_daily_pnl_path', '')}`")
            st.caption(f"Model Summary CSV: `{current_policy_result.get('policy_model_summary_path', '')}`")
            st.caption(f"Exit Summary CSV: `{current_policy_result.get('policy_exit_summary_path', '')}`")
            st.caption(f"Worst Trades CSV: `{current_policy_result.get('policy_worst_trades_path', '')}`")
            st.caption(f"Missed Risk CSV: `{current_policy_result.get('policy_dollar_missed_risk_path', '')}`")
            st.caption(
                f"Audit Assumptions CSV: `{current_policy_result.get('policy_audit_starting_assumptions_path', '')}`"
            )
            st.caption(
                f"Audit Daily Rows CSV: `{current_policy_result.get('policy_audit_first_20_daily_rows_path', '')}`"
            )
            st.caption(
                f"Audit Trades CSV: `{current_policy_result.get('policy_audit_first_20_trades_path', '')}`"
            )
            st.caption(
                f"Invariant Checks CSV: `{current_policy_result.get('policy_invariant_checks_path', '')}`"
            )
            st.caption(
                f"Profit/Loss Explanation CSV: `{current_policy_result.get('policy_profit_loss_explanation_path', '')}`"
            )

            runtime_profile = dict(current_policy_result.get("runtime_profile", {}))
            if runtime_profile:
                st.subheader("Runtime Profile")
                st.dataframe(
                    [{"metric": key, "value": value} for key, value in runtime_profile.items()],
                    width="stretch",
                    hide_index=True,
                )
            leaderboard = list(current_policy_result.get("model_economic_leaderboard", []))
            if leaderboard:
                st.subheader("Model Economic Leaderboard")
                st.dataframe(leaderboard[:20], width="stretch", hide_index=True)
            worst_trades = list(current_policy_result.get("worst_trades", []))
            if worst_trades:
                st.subheader("Worst 10 Trades")
                st.dataframe(worst_trades[:10], width="stretch", hide_index=True)
            best_trades = list(current_policy_result.get("best_trades", []))
            if best_trades:
                st.subheader("Best 10 Trades")
                st.dataframe(best_trades[:10], width="stretch", hide_index=True)
            invariant_checks = list(current_policy_result.get("policy_invariant_checks", []))
            if invariant_checks:
                st.subheader("Invariant Checks")
                st.dataframe(invariant_checks[:50], width="stretch", hide_index=True)
            profit_loss_explanation = list(current_policy_result.get("policy_profit_loss_explanation", []))
            if profit_loss_explanation:
                st.subheader("Why This Model Made/Lost Money")
                st.dataframe(profit_loss_explanation[:20], width="stretch", hide_index=True)
            audit_daily_preview = list(current_policy_result.get("policy_audit_daily_preview", []))
            if audit_daily_preview:
                st.subheader("Policy Mechanics Audit: First Daily Rows")
                st.dataframe(audit_daily_preview[:20], width="stretch", hide_index=True)
            audit_trades_preview = list(current_policy_result.get("policy_audit_trades_preview", []))
            if audit_trades_preview:
                st.subheader("Policy Mechanics Audit: First Trades")
                st.dataframe(audit_trades_preview[:20], width="stretch", hide_index=True)
            st.subheader("Portfolio Mechanics Check")
            mechanics_checks = _build_policy_mechanics_checks(
                policy_daily_path=str(current_policy_result.get("policy_daily_pnl_path", "")),
                policy_trades_path=str(current_policy_result.get("policy_trades_path", "")),
            )
            st.dataframe(mechanics_checks, width="stretch", hide_index=True)

    with ibkr_tab:
        st.subheader("Live IBKR Market Data Agent")
        default_ibkr_payload = load_json(DEFAULT_IBKR_INPUT)
        ibkr_col1, ibkr_col2, ibkr_col3 = st.columns(3)
        symbol = ibkr_col1.text_input("Symbol", value=str(default_ibkr_payload.get("symbol", "SPY")))
        host = ibkr_col2.text_input("Host", value=str(default_ibkr_payload.get("host", "127.0.0.1")))
        port = ibkr_col3.number_input("Port", value=int(default_ibkr_payload.get("port", 4001)), step=1)

        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        client_id = cfg_col1.number_input("Client ID", value=int(default_ibkr_payload.get("client_id", 73)), step=1)
        market_data_type = cfg_col2.number_input(
            "Market Data Type",
            value=int(default_ibkr_payload.get("market_data_type", 1)),
            step=1,
        )
        expiry_count = cfg_col3.number_input("Expiry Count", value=int(default_ibkr_payload.get("expiry_count", 2)), step=1)
        strike_count = cfg_col4.number_input("Strike Count", value=int(default_ibkr_payload.get("strike_count", 8)), step=1)

        opt_col1, opt_col2, opt_col3 = st.columns(3)
        exchange = opt_col1.text_input("Exchange", value=str(default_ibkr_payload.get("exchange", "SMART")))
        option_exchange = opt_col2.text_input(
            "Option Exchange",
            value=str(default_ibkr_payload.get("option_exchange", "SMART")),
        )
        currency = opt_col3.text_input("Currency", value=str(default_ibkr_payload.get("currency", "USD")))

        if st.button("Fetch IBKR Snapshot", type="primary"):
            payload = {
                "symbol": symbol.strip() or "SPY",
                "host": host.strip() or "127.0.0.1",
                "port": int(port),
                "client_id": int(client_id),
                "market_data_type": int(market_data_type),
                "exchange": exchange.strip() or "SMART",
                "option_exchange": option_exchange.strip() or "SMART",
                "currency": currency.strip() or "USD",
                "expiry_count": int(expiry_count),
                "strike_count": int(strike_count),
                "min_days_to_expiry": 0,
            }
            try:
                result = run_ibkr_market_data_agent(
                    input_payload=payload,
                    agent_path=default_ibkr_agent_path(),
                    storage_root=storage_root,
                    database_url=database_url or None,
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                _render_ibkr_result(st, result)

    with resume_tab:
        st.subheader("Resume a Review-Gated Daily Run")
        run_id = st.text_input("Run ID")
        decision = st.selectbox("Decision", ["approved", "rejected"])
        notes = st.text_area("Review Notes", height=120)
        if st.button("Resume Run"):
            try:
                result = resume_daily_regime_run(
                    run_id=run_id.strip(),
                    decision=decision,
                    notes=notes.strip() or None,
                    storage_root=storage_root,
                    database_url=database_url or None,
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                _render_daily_result(st, result, hmm_history=[])


if __name__ == "__main__":
    main()
