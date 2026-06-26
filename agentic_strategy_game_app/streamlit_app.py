"""Streamlit frontend for the strategy game app."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _ensure_streamlit_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "agentic_strategy_game_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_streamlit_imports()

from agentic_strategy_game_app.dashboard import (  # noqa: E402
    apply_market_force_overrides,
    company_metric_correlation_rows,
    company_rows,
    ecosystem_rows,
    load_scenario_world,
    market_force_rows,
    strategic_pressure_summary,
)
from agentic_strategy_game_app.app_runtime import run_vc_investor_agent  # noqa: E402
from agentic_strategy_game_app.contracts import VCPitchSessionState, WorldState  # noqa: E402
from agentic_strategy_game_app.engine import StrategyGameEngine  # noqa: E402
from agentic_strategy_game_app.player_strategy import interpret_player_strategy  # noqa: E402
from agentic_strategy_game_app.runtime_loop import synchronize_world_with_elapsed_time  # noqa: E402
from agentic_strategy_game_app.scenarios import list_scenarios  # noqa: E402
from agentic_strategy_game_app.simulation_clock import build_simulation_calendar_payload, simulated_date_label  # noqa: E402
from agentic_strategy_game_app.vc_pitch import (  # noqa: E402
    append_player_pitch_message,
    apply_vc_agent_response,
    build_vc_agent_input_payload,
    create_vc_pitch_session,
    extract_vc_agent_output_payload,
    parse_vc_agent_response,
)

CARD_LABEL_COLOR = "#facc15"
CARD_VALUE_COLOR = "#fef3c7"
CARD_CAPTION_COLOR = "#fde68a"
WIDGET_BORDER_COLOR = "#facc15"
WIDGET_BORDER_FOCUS = "#fde68a"


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install it with `pip install streamlit` "
            "or add it to the app environment first."
        ) from exc
    return st


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _render_simulation_calendar_card(st, *, period_label: str, anchor_epoch: float, now_epoch: float) -> None:
    payload = build_simulation_calendar_payload(period_label).to_dict()
    current_date = simulated_date_label(period_label, anchor_epoch=anchor_epoch, now_epoch=now_epoch)
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 0.9rem;
            padding: 0.85rem 1rem;
            background: linear-gradient(180deg, rgba(250,250,250,0.03), rgba(250,250,250,0.01));
            min-height: 5.75rem;
            text-align: right;
        ">
            <div style="font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em; color:{CARD_LABEL_COLOR};">Simulation Calendar</div>
            <div style="font-size:1.15rem; font-weight:700; margin-top:0.35rem; color:{CARD_VALUE_COLOR};">{current_date}</div>
            <div style="font-size:0.92rem; margin-top:0.2rem; color:{CARD_VALUE_COLOR};">{payload["quarter_label"]}</div>
            <div style="font-size:0.76rem; margin-top:0.35rem; color:{CARD_CAPTION_COLOR};">1 day every 5 seconds until {payload["end_label"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_card(st, *, label: str, value: str, caption: str = "") -> None:
    caption_html = (
        f'<div style="font-size:0.82rem; color:{CARD_CAPTION_COLOR}; margin-top:0.45rem;">{caption}</div>'
        if caption
        else ""
    )
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 0.9rem;
            padding: 0.95rem 1rem;
            min-height: 6rem;
            background: linear-gradient(180deg, rgba(250,250,250,0.03), rgba(250,250,250,0.01));
        ">
            <div style="font-size:0.76rem; text-transform:uppercase; letter-spacing:0.05em; color:{CARD_LABEL_COLOR}; margin-bottom:0.35rem;">{label}</div>
            <div style="font-size:1.05rem; font-weight:650; line-height:1.25; color:{CARD_VALUE_COLOR};">{value}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tone(value: float) -> tuple[str, str]:
    if value >= 0.7:
        return "#b42318", "#fef3f2"
    if value >= 0.5:
        return "#b54708", "#fffaeb"
    return "#027a48", "#ecfdf3"


def _render_pressure_cards(st, summary: dict[str, float]) -> None:
    st.subheader("Strategic Pressure Map")
    columns = st.columns(len(summary))
    for column, (label, value) in zip(columns, summary.items()):
        fg, bg = _tone(value)
        column.markdown(
            f"""
            <div style="
                border: 1px solid rgba(128,128,128,0.22);
                border-radius: 0.9rem;
                padding: 0.95rem 1rem;
                min-height: 6rem;
                background: rgba(255,255,255,0.02);
            ">
                <div style="font-size:0.76rem; text-transform:uppercase; letter-spacing:0.05em; color:{CARD_LABEL_COLOR}; margin-bottom:0.45rem;">{label.replace('_', ' ')}</div>
                <div>
                    <span style="background:{bg}; color:{fg}; padding:0.28rem 0.58rem; border-radius:999px; font-weight:700;">{value:.3f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "These are derived pressure indicators from the current market-force snapshot. "
        "They are not learned scores yet."
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


def _render_turn_result(st, *, turn_result: dict | None) -> None:
    st.subheader("Latest Turn Result")
    if not turn_result:
        st.info("No turn has been run yet. Interpret a player strategy and then execute a turn.")
        return

    top_col1, top_col2, top_col3 = st.columns(3)
    _render_summary_card(top_col1, label="Executed Turn", value=f"{turn_result['period_label']} -> Turn {turn_result['turn_number']}")
    _render_summary_card(top_col2, label="Accepted Actions", value=str(len(turn_result.get("actions_accepted", []))))
    _render_summary_card(top_col3, label="Rejected Actions", value=str(len(turn_result.get("actions_rejected", []))))
    st.caption(str(turn_result.get("narrative_summary", "")))

    accepted = list(turn_result.get("actions_accepted", []))
    if accepted:
        st.markdown("**Accepted Actions**")
        st.dataframe(
            [
                {
                    "actor": item.get("actor_id"),
                    "action_type": item.get("action_type"),
                    "intensity": item.get("intensity"),
                    "resource_cost": item.get("resource_cost"),
                }
                for item in accepted
            ],
            use_container_width=True,
            hide_index=True,
        )

    rejected = list(turn_result.get("actions_rejected", []))
    if rejected:
        st.markdown("**Rejected Actions**")
        st.dataframe(
            [
                {
                    "actor": item.get("action", {}).get("actor_id"),
                    "action_type": item.get("action", {}).get("action_type"),
                    "reason": item.get("reason"),
                }
                for item in rejected
            ],
            use_container_width=True,
            hide_index=True,
        )

    passive_dynamics = dict(turn_result.get("market_results", {})).get("passive_dynamics", {})
    if passive_dynamics:
        st.markdown("**Ecosystem Drift**")
        st.dataframe(
            [
                {
                    "company": company_id,
                    "market_share_delta": metrics.get("market_share_delta"),
                    "momentum_delta": metrics.get("strategic_momentum_delta"),
                }
                for company_id, metrics in passive_dynamics.items()
            ],
            use_container_width=True,
            hide_index=True,
        )


def _run_vc_pitch_round(
    session: VCPitchSessionState,
    *,
    max_rounds: int,
) -> tuple[VCPitchSessionState, dict[str, object]]:
    payload = build_vc_agent_input_payload(session, max_rounds=max_rounds)
    result = run_vc_investor_agent(
        input_payload=payload,
        storage_root=Path("agentic_strategy_game_app/.workflow_memory"),
    )
    raw_response = extract_vc_agent_output_payload(result)
    parsed = parse_vc_agent_response(raw_response)
    return apply_vc_agent_response(session, parsed), result


def _render_vc_transcript(st, session: VCPitchSessionState) -> None:
    st.markdown("**Pitch Transcript**")
    if not session.transcript:
        st.caption("No VC conversation has started yet.")
        return
    for turn in session.transcript:
        if turn.speaker == "player":
            label = "Founder"
            accent = "rgba(31, 111, 235, 0.08)"
            border = "rgba(31, 111, 235, 0.28)"
        else:
            label = "VC"
            accent = "rgba(180, 83, 9, 0.08)"
            border = "rgba(180, 83, 9, 0.28)"
        st.markdown(
            f"""
            <div style="
                border: 1px solid {border};
                border-radius: 0.85rem;
                padding: 0.85rem 1rem;
                margin-bottom: 0.65rem;
                background: {accent};
            ">
                <div style="font-size:0.76rem; text-transform:uppercase; letter-spacing:0.05em; color:rgba(105,105,105,0.95); margin-bottom:0.45rem;">
                    {label} · Round {turn.round_number}
                </div>
                <div style="font-size:0.96rem; line-height:1.45; white-space:pre-wrap;">{turn.content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_vc_decision(st, session: VCPitchSessionState) -> None:
    latest = session.latest_agent_response
    if latest is None:
        return
    decision = latest.decision
    signal_col, focus_col = st.columns([1.2, 1.8])
    _render_summary_card(
        signal_col,
        label="VC Signal",
        value=latest.tentative_signal.replace("_", " ").title(),
        caption=f"Session status: {session.status.replace('_', ' ')}",
    )
    focus_text = ", ".join(latest.diligence_focus) if latest.diligence_focus else "No explicit diligence focus logged."
    _render_summary_card(
        focus_col,
        label="Diligence Focus",
        value=focus_text,
    )
    if decision is None:
        if latest.followup_questions:
            st.markdown("**Current VC Questions**")
            for question in latest.followup_questions:
                st.markdown(f"- {question}")
        return
    st.markdown("**Investment Decision**")
    decision_cols = st.columns(4)
    _render_summary_card(decision_cols[0], label="Outcome", value=decision.outcome.replace("_", " ").title())
    _render_summary_card(decision_cols[1], label="Amount Offered", value=f"${decision.amount_offered:,.0f}")
    _render_summary_card(decision_cols[2], label="Equity Requested", value=f"{decision.equity_requested * 100:.1f}%")
    _render_summary_card(decision_cols[3], label="Founder Ask", value=f"${session.capital_requested:,.0f} for {session.equity_offered * 100:.1f}%")
    st.caption(decision.rationale or latest.summary)
    if decision.terms:
        st.markdown("**Terms And Conditions**")
        for term in decision.terms:
            st.markdown(f"- {term}")


def _render_player_strategy(st, world, *, max_actions: int):
    st.subheader("Player Strategy")
    company_options = list(world.companies)
    default_actor_index = company_options.index("incumbent_platform") if "incumbent_platform" in company_options else 0
    player_actor_id = st.selectbox(
        "Playable Company",
        options=company_options,
        index=default_actor_index,
        format_func=lambda value: world.companies[value].name,
        key="player_actor_id",
    )
    strategy_text = st.text_area(
        "Describe your strategy in plain English",
        value=(
            "Invest heavily in AI, keep pricing disciplined, and expand marketing so we can defend "
            "our position without looking asleep."
        ),
        height=140,
        key="player_strategy_text",
    )
    interpret_now = st.button("Interpret Strategy", type="primary", key="interpret_player_strategy")

    if not strategy_text.strip():
        st.caption(
            "This is the human-in-the-middle layer: you specify strategy in plain English, "
            "and the user actor agent converts it into structured action proposals."
        )
        return None, False

    try:
        interpretation = interpret_player_strategy(
            world=world,
            actor_id=player_actor_id,
            raw_strategy=strategy_text,
            max_actions=max_actions,
        )
    except Exception as exc:
        st.error(str(exc))
        return None, False

    if not interpret_now:
        st.caption(
            "This is the human-in-the-middle layer: you specify strategy in plain English, "
            "and the user actor agent converts it into structured action proposals."
        )

    subject_col1, subject_col2, subject_col3 = st.columns(3)
    _render_summary_card(subject_col1, label="Player Company", value=interpretation.actor_name)
    _render_summary_card(subject_col2, label="Actions Interpreted", value=str(len(interpretation.actions)))
    _render_summary_card(
        subject_col3,
        label="Warnings",
        value=str(len(interpretation.warnings)),
        caption="Deterministic parser warnings",
    )

    st.caption(interpretation.summary)

    if interpretation.interpreted_goals:
        st.markdown("**Interpreted Goals**")
        for item in interpretation.interpreted_goals:
            st.markdown(f"- {item}")

    if interpretation.actions:
        st.markdown("**Structured Action Proposals**")
        st.dataframe(
            [
                {
                    "action_type": action.action_type,
                    "intensity": action.intensity,
                    "resource_cost": action.resource_cost,
                    "goal": action.expected_effects.get("goal"),
                    "rationale": action.rationale,
                }
                for action in interpretation.actions
            ],
            use_container_width=True,
            hide_index=True,
        )

    systems_col1, systems_col2 = st.columns(2)
    with systems_col1:
        st.markdown("**Internal Systems Impacted**")
        if interpretation.internal_systems:
            for item in interpretation.internal_systems:
                st.markdown(f"- {item}")
        else:
            st.caption("No internal systems inferred.")
    with systems_col2:
        st.markdown("**External Systems Impacted**")
        if interpretation.external_systems:
            for item in interpretation.external_systems:
                st.markdown(f"- {item}")
        else:
            st.caption("No external systems inferred.")

    if interpretation.warnings:
        st.markdown("**Interpreter Warnings**")
        for item in interpretation.warnings:
            st.warning(item)

    with st.expander("Interpreted Strategy JSON"):
        st.code(_pretty_json(interpretation.to_dict()), language="json")

    should_run_turn = st.button("Run Player Turn", type="secondary", key="run_player_turn")
    return interpretation, should_run_turn


def _render_vc_pitch_panel(
    st,
    *,
    world: WorldState,
    state_vc_session_key: str,
    state_vc_result_key: str,
) -> None:
    st.subheader("VC Raise")
    st.caption(
        "This is the first non-player LLM-backed agent. The founder pitches, the investor grills on growth, projections, "
        "evidence, and market realism, then decides or negotiates."
    )
    current_session_payload = st.session_state.get(state_vc_session_key)
    current_session = (
        VCPitchSessionState.from_dict(current_session_payload)
        if isinstance(current_session_payload, dict)
        else None
    )

    controls_col, transcript_col = st.columns([1.1, 1.45])
    with controls_col:
        if current_session is None:
            company_options = list(world.companies)
            default_actor_index = company_options.index("ai_native_startup") if "ai_native_startup" in company_options else 0
            actor_id = st.selectbox(
                "Pitching Company",
                options=company_options,
                index=default_actor_index,
                format_func=lambda value: world.companies[value].name,
                key="vc_pitch_actor_id",
            )
            capital_requested = st.number_input(
                "Capital Requested",
                min_value=250000.0,
                max_value=100_000_000.0,
                value=2_500_000.0,
                step=250_000.0,
                format="%.0f",
                key="vc_capital_requested",
            )
            equity_offered_percent = st.slider(
                "Equity Offered (%)",
                min_value=1.0,
                max_value=49.0,
                value=12.0,
                step=0.5,
                key="vc_equity_offered_percent",
            )
            max_rounds = st.slider(
                "Max VC Rounds Before Decision",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
                key="vc_max_rounds",
            )
            strategy_summary = st.text_area(
                "Fundraising Strategy",
                value=(
                    "We want to raise capital to accelerate productized AI workflows, expand enterprise GTM, "
                    "and turn early adoption into a repeatable revenue engine."
                ),
                height=160,
                key="vc_strategy_summary",
            )
            start_pitch = st.button("Start VC Pitch", type="primary", key="start_vc_pitch")
            st.caption("Requires an LLM-capable `agentic_harness` environment, for example `OPENAI_API_KEY` in your shell.")
            if start_pitch:
                try:
                    session = create_vc_pitch_session(
                        world=world,
                        actor_id=actor_id,
                        capital_requested=float(capital_requested),
                        equity_offered=float(equity_offered_percent) / 100.0,
                        strategy_summary=strategy_summary,
                    )
                    opening_message = (
                        f"We are raising ${capital_requested:,.0f} for {equity_offered_percent:.1f}% of the business. "
                        f"Our strategy is: {strategy_summary}"
                    )
                    session = append_player_pitch_message(session, opening_message)
                    session, result = _run_vc_pitch_round(session, max_rounds=int(max_rounds))
                    st.session_state[state_vc_session_key] = session.to_dict()
                    st.session_state[state_vc_result_key] = result
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            latest = current_session.latest_agent_response
            top_cols = st.columns(3)
            _render_summary_card(top_cols[0], label="Pitching Company", value=current_session.company_name)
            _render_summary_card(top_cols[1], label="Capital Requested", value=f"${current_session.capital_requested:,.0f}")
            _render_summary_card(top_cols[2], label="Equity Offered", value=f"{current_session.equity_offered * 100:.1f}%")
            _render_vc_decision(st, current_session)
            if latest and latest.mode != "decision":
                st.markdown("**Founder Response**")
                answer = st.text_area(
                    "Answer the investor's latest questions",
                    height=180,
                    key="vc_founder_answer",
                )
                action_cols = st.columns([1.0, 1.0, 2.0])
                submit_answer = action_cols[0].button("Submit Answer", type="primary", key="submit_vc_answer")
                reset_pitch = action_cols[1].button("Reset VC Session", key="reset_vc_session_active")
                if submit_answer:
                    if not answer.strip():
                        st.warning("Enter a founder response before submitting.")
                    else:
                        try:
                            max_rounds = int(st.session_state.get("vc_max_rounds", 3))
                            updated = append_player_pitch_message(current_session, answer)
                            updated, result = _run_vc_pitch_round(updated, max_rounds=max_rounds)
                            st.session_state[state_vc_session_key] = updated.to_dict()
                            st.session_state[state_vc_result_key] = result
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                if reset_pitch:
                    st.session_state.pop(state_vc_session_key, None)
                    st.session_state.pop(state_vc_result_key, None)
                    st.session_state.pop("vc_founder_answer", None)
                    st.rerun()
            else:
                if st.button("Reset VC Session", key="reset_vc_session_done"):
                    st.session_state.pop(state_vc_session_key, None)
                    st.session_state.pop(state_vc_result_key, None)
                    st.session_state.pop("vc_founder_answer", None)
                    st.rerun()

    with transcript_col:
        if current_session is None:
            st.info("Start a VC pitch to open the investor conversation, diligence questions, and eventual term sheet decision.")
        else:
            _render_vc_transcript(st, current_session)
            with st.expander("VC Session JSON"):
                st.code(_pretty_json(current_session.to_dict()), language="json")
            latest = current_session.latest_agent_response
            if latest is not None and latest.raw_response.strip():
                with st.expander("Raw VC Agent Response"):
                    st.code(latest.raw_response, language="json")


def _render_clock_fragment(
    st,
    *,
    state_world_key: str,
    state_history_key: str,
    state_latest_turn_key: str,
    state_clock_anchor_key: str,
    state_notice_key: str,
    max_actions_per_turn: int,
) -> None:
    def _fragment_body() -> None:
        if state_world_key not in st.session_state:
            return
        engine = StrategyGameEngine(max_actions_per_turn=max_actions_per_turn)
        world = WorldState.from_dict(st.session_state[state_world_key])
        anchor_epoch = float(st.session_state.get(state_clock_anchor_key, time.time()))
        now_epoch = time.time()
        synced_world, synced_anchor, generated_turns = synchronize_world_with_elapsed_time(
            world=world,
            clock_anchor_epoch=anchor_epoch,
            now_epoch=now_epoch,
            engine=engine,
        )
        if generated_turns:
            st.session_state[state_world_key] = synced_world.to_dict()
            turn_history = list(st.session_state.get(state_history_key, []))
            turn_history.extend(turn.to_dict() for turn in generated_turns)
            st.session_state[state_history_key] = turn_history
            st.session_state[state_latest_turn_key] = generated_turns[-1].to_dict()
            st.session_state[state_clock_anchor_key] = synced_anchor
            st.session_state[state_notice_key] = (
                f"The quarter rolled forward automatically to {synced_world.current_period_label}."
            )
            st.rerun()
        _render_simulation_calendar_card(
            st,
            period_label=world.current_period_label,
            anchor_epoch=anchor_epoch,
            now_epoch=now_epoch,
        )

    if hasattr(st, "fragment"):
        @st.fragment(run_every="1s")
        def _clock_fragment() -> None:
            _fragment_body()

        _clock_fragment()
    else:
        _fragment_body()


def main() -> None:
    st = _load_streamlit()

    st.set_page_config(
        page_title="Agentic Strategy Game App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_widget_theme(st)
    title_col, calendar_col = st.columns([4.3, 1.7])

    scenario_name = "b2b_saas_ai_disruption"
    config, base_world = load_scenario_world(scenario_name)

    with st.sidebar:
        st.header("Scenario Controls")
        selected_scenario = st.selectbox("Scenario", options=list_scenarios(), index=0)
        if selected_scenario != scenario_name:
            config, base_world = load_scenario_world(selected_scenario)
            scenario_name = selected_scenario
        st.caption("These sliders define the baseline market state. Use reset to apply them to the simulation.")
        max_actions_per_turn = st.number_input(
            "Max Actions Per Turn",
            min_value=1,
            max_value=5,
            value=int(config.get("max_actions_per_turn", 2)),
            step=1,
        )
        reset_simulation = st.button("Reset Simulation State", key="reset_strategy_game_state")
        overrides = {
            "growth_rate": st.slider("Growth Rate", min_value=-0.20, max_value=0.50, value=float(base_world.market_forces.growth_rate), step=0.01),
            "rivalry_intensity": st.slider("Rivalry Intensity", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.rivalry_intensity), step=0.01),
            "regulatory_pressure": st.slider("Regulatory Pressure", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.regulatory_pressure), step=0.01),
            "capital_availability": st.slider("Capital Availability", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.capital_availability), step=0.01),
            "talent_availability": st.slider("Talent Availability", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.talent_availability), step=0.01),
            "technology_shift_intensity": st.slider("Technology Shift Intensity", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.technology_shift_intensity), step=0.01),
            "macroeconomic_pressure": st.slider("Macroeconomic Pressure", min_value=0.0, max_value=1.0, value=float(base_world.market_forces.macroeconomic_pressure), step=0.01),
        }

    state_scenario_key = "strategy_game_selected_scenario"
    state_world_key = "strategy_game_current_world"
    state_history_key = "strategy_game_turn_history"
    state_latest_turn_key = "strategy_game_latest_turn"
    state_clock_anchor_key = "strategy_game_clock_anchor_epoch"
    state_notice_key = "strategy_game_clock_notice"
    state_vc_session_key = "strategy_game_vc_session"
    state_vc_result_key = "strategy_game_vc_result"

    if (
        reset_simulation
        or state_world_key not in st.session_state
        or st.session_state.get(state_scenario_key) != scenario_name
    ):
        initialized_world = apply_market_force_overrides(base_world, overrides)
        st.session_state[state_scenario_key] = scenario_name
        st.session_state[state_world_key] = initialized_world.to_dict()
        st.session_state[state_history_key] = []
        st.session_state[state_latest_turn_key] = None
        st.session_state[state_clock_anchor_key] = time.time()
        st.session_state[state_notice_key] = None
        st.session_state.pop(state_vc_session_key, None)
        st.session_state.pop(state_vc_result_key, None)
        st.session_state.pop("vc_founder_answer", None)

    world = WorldState.from_dict(st.session_state[state_world_key])
    pressure_summary = strategic_pressure_summary(world.market_forces)
    engine = StrategyGameEngine(max_actions_per_turn=int(max_actions_per_turn))

    interpretation, should_run_turn = _render_player_strategy(st, world, max_actions=int(max_actions_per_turn))
    if should_run_turn and interpretation is not None:
        next_world, turn_result = engine.run_turn(world, interpretation.actions, advance_time=False)
        st.session_state[state_world_key] = next_world.to_dict()
        turn_history = list(st.session_state.get(state_history_key, []))
        turn_history.append(turn_result.to_dict())
        st.session_state[state_history_key] = turn_history
        st.session_state[state_latest_turn_key] = turn_result.to_dict()
        st.session_state[state_notice_key] = None
        world = next_world
        pressure_summary = strategic_pressure_summary(world.market_forces)
        st.success(f"Executed an in-quarter decision during {world.current_period_label}. The simulation clock did not jump.")
        st.rerun()

    world = WorldState.from_dict(st.session_state[state_world_key])
    pressure_summary = strategic_pressure_summary(world.market_forces)
    notice_text = st.session_state.get(state_notice_key)

    with title_col:
        st.title("Agentic Strategy Game App")
        st.caption(
            "Scenario explorer for the business strategy simulator built on top of agentic_harness. "
            "One turn equals one business quarter; the calendar shows time progressing inside the active quarter."
        )
        if notice_text:
            st.info(str(notice_text))
            st.session_state[state_notice_key] = None
    with calendar_col:
        _render_clock_fragment(
            st,
            state_world_key=state_world_key,
            state_history_key=state_history_key,
            state_latest_turn_key=state_latest_turn_key,
            state_clock_anchor_key=state_clock_anchor_key,
            state_notice_key=state_notice_key,
            max_actions_per_turn=int(max_actions_per_turn),
        )

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)
    _render_summary_card(
        top_col1,
        label="Scenario",
        value=scenario_name.replace("_", " ").title(),
        caption="Initial scenario scaffold",
    )
    _render_summary_card(
        top_col2,
        label="Turn",
        value=f"{world.current_period_label} (Turn {world.current_turn})",
        caption="Current simulation decision point",
    )
    _render_summary_card(
        top_col3,
        label="Companies",
        value=str(len(world.companies)),
        caption="Firms with full internal state",
    )
    _render_summary_card(
        top_col4,
        label="Ecosystem Actors",
        value=str(len(world.ecosystem_agents)),
        caption="Firms plus market actors",
    )

    _render_pressure_cards(st, pressure_summary)
    _render_turn_result(st, turn_result=st.session_state.get(state_latest_turn_key))

    overview_tab, companies_tab, ecosystem_tab, vc_raise_tab, turn_log_tab, diagnostics_tab = st.tabs(
        ["Market Forces", "Companies", "Ecosystem", "VC Raise", "Turn Log", "Diagnostics"]
    )

    with overview_tab:
        st.subheader("External Forces")
        st.dataframe(market_force_rows(world), use_container_width=True, hide_index=True)
        st.markdown("**Strategic Tensions**")
        for item in list(world.metadata.get("initial_questions", [])):
            st.markdown(f"- {item}")

    with companies_tab:
        st.subheader("Company Comparison")
        st.dataframe(company_rows(world), use_container_width=True, hide_index=True)
        st.caption(
            "These metrics are the initial internal states that the future simulation engine will update over time."
        )

    with ecosystem_tab:
        st.subheader("Ecosystem Actors")
        st.dataframe(ecosystem_rows(world), use_container_width=True, hide_index=True)
        st.caption(
            "These profiles define strategic personalities and incentives for company and non-company actors."
        )

    with vc_raise_tab:
        _render_vc_pitch_panel(
            st,
            world=world,
            state_vc_session_key=state_vc_session_key,
            state_vc_result_key=state_vc_result_key,
        )

    with turn_log_tab:
        st.subheader("Turn History")
        turn_history = list(st.session_state.get(state_history_key, []))
        if turn_history:
            st.dataframe(
                [
                    {
                        "turn": item.get("turn_number"),
                        "period": item.get("period_label"),
                        "accepted_actions": len(item.get("actions_accepted", [])),
                        "rejected_actions": len(item.get("actions_rejected", [])),
                        "summary": item.get("narrative_summary"),
                    }
                    for item in turn_history
                ],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Turn History JSON"):
                st.code(_pretty_json(turn_history), language="json")
        else:
            st.info("No turns have been executed yet.")

    with diagnostics_tab:
        st.subheader("Correlation View")
        st.dataframe(company_metric_correlation_rows(world), use_container_width=True, hide_index=True)
        st.caption(
            "This is a cross-company metric correlation matrix for the current initial snapshot, "
            "not a historical time-series correlation yet."
        )
        st.subheader("Config")
        st.code(_pretty_json(config), language="json")
        st.subheader("World State")
        st.code(_pretty_json(world.to_dict()), language="json")


if __name__ == "__main__":
    main()
