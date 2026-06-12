"""Streamlit frontend for the volatility regime application."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


def _ensure_streamlit_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "agentic_vol_regime_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_streamlit_imports()

from agentic_vol_regime_app import (  # noqa: E402
    default_agent_path,
    default_hmm_agent_path,
    default_hmm_v2_agent_path,
    default_hmm_v3_agent_path,
    default_ibkr_agent_path,
    default_ml_agent_path,
    load_or_run_historical_belief_report,
    load_latest_live_daily_observation,
    load_recent_hmm_state_history,
    reset_hmm_persisted_state,
    snapshot_hmm_baseline,
    resume_daily_regime_run,
    run_daily_regime_agent,
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
CARD_LABEL_COLOR = "#facc15"
CARD_VALUE_COLOR = "#fef3c7"
WIDGET_BORDER_COLOR = "#facc15"
WIDGET_BORDER_FOCUS = "#fde68a"
REGIME_ENGINE_OPTIONS = ["Heuristic Agent", "ML Agent", "HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent"]


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
    return default_agent_path()


def _default_history_days_for_agent_choice(choice: str, fallback: int) -> int:
    return 756 if choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent"} else int(fallback)


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
        st.dataframe(rows, use_container_width=True, hide_index=True)


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
        st.dataframe(rows, use_container_width=True, hide_index=True)


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
        st.dataframe(step_rows, use_container_width=True, hide_index=True)


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
            use_container_width=True,
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
        st.dataframe(rows, use_container_width=True, hide_index=True)
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

    daily_tab, historical_tab, ibkr_tab, resume_tab = st.tabs(
        ["Daily Belief Report", "Historical Belief Reports", "IBKR Snapshot Agent", "Resume Daily Review"]
    )
    state_daily_result_key = "vol_regime_last_daily_result"
    state_historical_report_key = "vol_regime_historical_report_result"

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
        enable_as_of_date = st.checkbox(
            "Run as prior as-of date",
            value=False,
            help=(
                "Rewinds the loaded snapshot history to the selected business date and runs the "
                "report as if that were the observation date."
            ),
        )
        selected_as_of_date = (
            st.date_input(
                "As-of Date",
                value=date.today(),
                max_value=date.today(),
                disabled=not enable_as_of_date,
            )
            if enable_as_of_date
            else None
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
            if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent"}:
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

        if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent"}:
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
                if enable_as_of_date and selected_as_of_date is not None:
                    payload["as_of_date"] = selected_as_of_date.isoformat()
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
                if agent_choice in {"HMMv1 Agent", "HMMv2 Agent", "HMMv3 Agent"}
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
