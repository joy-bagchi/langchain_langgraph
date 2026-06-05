"""Streamlit frontend for the volatility regime application."""

from __future__ import annotations

import json
import sys
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
    default_ibkr_agent_path,
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


def _pretty_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_json_text(value: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not value.strip():
        return dict(fallback)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a top-level JSON object.")
    return parsed


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


def _render_daily_result(st, result: dict[str, Any]) -> None:
    outputs = dict(result.get("named_outputs", {}))
    daily_report = dict(outputs.get("daily_report", {}))
    belief_state = dict(outputs.get("belief_state", {}))
    alert_record = dict(outputs.get("alert_record", {}))
    policy = dict(outputs.get("policy_recommendation", {}))

    _render_runtime_diagnostics(st, result)

    st.subheader("Run Summary")
    status_col, regime_col, alert_col, action_col = st.columns(4)
    status_col.metric("Status", str(result.get("status", "")))
    regime_col.metric(
        "Top Regime",
        max(dict(belief_state.get("beliefs", {})), key=dict(belief_state.get("beliefs", {})).get)
        if belief_state.get("beliefs")
        else "n/a",
    )
    alert_col.metric("Alert", str(alert_record.get("severity", "n/a")))
    action_col.metric("Action", str(policy.get("recommended_action", "n/a")))

    if daily_report.get("markdown"):
        st.subheader("Report")
        st.markdown(str(daily_report["markdown"]))

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


def main() -> None:
    st = _load_streamlit()

    st.set_page_config(
        page_title="Agentic Vol Regime App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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

    daily_tab, ibkr_tab, resume_tab = st.tabs(
        ["Daily Belief Report", "IBKR Snapshot Agent", "Resume Daily Review"]
    )

    with daily_tab:
        st.subheader("Deterministic Daily Workflow")
        default_daily_payload = load_json(DEFAULT_DAILY_INPUT)
        default_live_payload = load_json(DEFAULT_DAILY_LIVE_INPUT)
        use_live_ibkr = st.checkbox(
            "Refresh regime inputs from live IBKR",
            value=True,
            help=(
                "Keeps the original Daily Belief Report behavior, but refreshes SPY, VIX, "
                "VVIX, VIX9D, VIX3M, history, and the SPY option chain from IBKR before the "
                "workflow computes the report."
            ),
        )

        effective_daily_payload: dict[str, Any]
        if use_live_ibkr:
            ibkr_defaults = dict(default_live_payload.get("ibkr", {}))
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
                    value=int(ibkr_defaults.get("history_days", 30)),
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
        else:
            daily_json = st.text_area(
                "Daily Input JSON",
                value=_pretty_json(default_daily_payload),
                height=360,
                key="daily_json",
            )
            effective_daily_payload = _parse_json_text(daily_json, fallback=default_daily_payload)

        if st.button("Run Daily Workflow", type="primary"):
            try:
                payload = dict(effective_daily_payload)
                result = run_daily_regime_agent(
                    input_payload=payload,
                    storage_root=storage_root,
                    database_url=database_url or None,
                    langsmith_tracing=langsmith_tracing,
                    langsmith_project=langsmith_project or None,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                _render_daily_result(st, result)

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
                _render_daily_result(st, result)


if __name__ == "__main__":
    main()
