"""Streamlit frontend for the IBKR account dashboard app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_streamlit_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "agentic_ibkr_account_app"
    for candidate in (str(repo_root), str(package_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


_ensure_streamlit_imports()

from agentic_ibkr_account_app.app_runtime import fetch_ibkr_account_snapshot  # noqa: E402


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install it with `pip install streamlit` "
            "or `pip install .[ui]` from agentic_ibkr_account_app."
        ) from exc
    return st


CARD_LABEL_COLOR = "#facc15"
CARD_VALUE_COLOR = "#fef3c7"
WIDGET_BORDER_COLOR = "#facc15"
WIDGET_BORDER_FOCUS = "#fde68a"


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
        .stNumberInput input {{
            border-color: {WIDGET_BORDER_COLOR} !important;
        }}
        .stTextInput input:hover,
        .stNumberInput input:hover {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
        }}
        .stTextInput input:focus,
        .stNumberInput input:focus {{
            border-color: {WIDGET_BORDER_FOCUS} !important;
            box-shadow: 0 0 0 1px {WIDGET_BORDER_FOCUS} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    text = str(value).strip()
    return text or "n/a"


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _render_dashboard(st, snapshot: dict[str, Any]) -> None:
    dashboard = dict(snapshot.get("dashboard", {}))
    row1 = st.columns(4)
    _render_summary_card(row1[0], label="Net Liquidation", value=_format_metric_value(dashboard.get("net_liquidation")))
    _render_summary_card(row1[1], label="Available Funds", value=_format_metric_value(dashboard.get("available_funds")))
    _render_summary_card(row1[2], label="Buying Power", value=_format_metric_value(dashboard.get("buying_power")))
    _render_summary_card(row1[3], label="Gross Position Value", value=_format_metric_value(dashboard.get("gross_position_value")))

    row2 = st.columns(4)
    _render_summary_card(row2[0], label="Position Count", value=_format_metric_value(dashboard.get("position_count")))
    _render_summary_card(row2[1], label="Open Orders", value=_format_metric_value(dashboard.get("open_order_count")))
    _render_summary_card(row2[2], label="Recent Transactions", value=_format_metric_value(dashboard.get("recent_transaction_count")))
    _render_summary_card(row2[3], label="Positions Unrealized PnL", value=_format_metric_value(dashboard.get("positions_unrealized_pnl")))

    row3 = st.columns(4)
    _render_summary_card(row3[0], label="Excess Liquidity", value=_format_metric_value(dashboard.get("excess_liquidity")))
    _render_summary_card(row3[1], label="Initial Margin", value=_format_metric_value(dashboard.get("init_margin_req")))
    _render_summary_card(row3[2], label="Maintenance Margin", value=_format_metric_value(dashboard.get("maint_margin_req")))
    _render_summary_card(row3[3], label="Positions Realized PnL", value=_format_metric_value(dashboard.get("positions_realized_pnl")))

    positions_df = _frame(list(snapshot.get("positions", [])))
    if not positions_df.empty:
        st.subheader("Top Positions by Market Value")
        if "market_value" in positions_df.columns:
            top_positions = positions_df.copy()
            top_positions["market_value_abs"] = pd.to_numeric(top_positions["market_value"], errors="coerce").abs()
            top_positions = top_positions.sort_values("market_value_abs", ascending=False).drop(columns=["market_value_abs"])
            st.dataframe(top_positions.head(10), width="stretch", hide_index=True)

    summary_df = _frame(list(snapshot.get("account_summary", [])))
    if not summary_df.empty:
        st.subheader("Account Summary")
        st.dataframe(summary_df, width="stretch", hide_index=True)


def main() -> None:
    st = _load_streamlit()
    st.set_page_config(page_title="IBKR Account Dashboard", page_icon=":bar_chart:", layout="wide")
    _inject_widget_theme(st)

    st.title("IBKR Account Dashboard")
    st.caption("Dashboard-first account monitor on top of the local agentic harness runtime.")

    with st.sidebar:
        st.header("Connection")
        host = st.text_input("Host", value="127.0.0.1")
        port = st.number_input("Port", value=4001, step=1)
        client_id = st.number_input("Client ID", value=91, step=1)
        readonly = st.checkbox("Readonly", value=True)
        max_fills = st.number_input("Max Transactions", value=100, min_value=10, max_value=1000, step=10)
        max_orders = st.number_input("Max Orders/Trades", value=100, min_value=10, max_value=1000, step=10)
        discovery_button = st.button("Discover Accounts")
        accounts_state_key = "agentic_ibkr_account_accounts"
        if accounts_state_key not in st.session_state:
            st.session_state[accounts_state_key] = []
        discovered_accounts = list(st.session_state.get(accounts_state_key, []))
        account_options = discovered_accounts if discovered_accounts else [""]
        default_account_index = 0
        account = st.selectbox(
            "Account",
            options=account_options,
            index=default_account_index,
            format_func=lambda value: value or "Auto-detect from IBKR",
        )
        fetch_button = st.button("Fetch Account Snapshot", type="primary")

    state_key = "agentic_ibkr_account_snapshot"
    accounts_state_key = "agentic_ibkr_account_accounts"
    if discovery_button:
        discovery_payload = {
            "host": host.strip() or "127.0.0.1",
            "port": int(port),
            "client_id": int(client_id),
            "account": None,
            "readonly": bool(readonly),
            "max_fills": 10,
            "max_orders": 10,
            "include_completed_orders": False,
        }
        try:
            discovered_snapshot = fetch_ibkr_account_snapshot(input_payload=discovery_payload)
        except Exception as exc:
            st.error(str(exc))
        else:
            managed_accounts = list(discovered_snapshot.get("managed_accounts", []))
            st.session_state[accounts_state_key] = managed_accounts
            if managed_accounts:
                st.session_state[state_key] = discovered_snapshot
            else:
                st.warning("IBKR did not return any managed accounts.")

    if fetch_button:
        payload = {
            "host": host.strip() or "127.0.0.1",
            "port": int(port),
            "client_id": int(client_id),
            "account": str(account).strip() or None,
            "readonly": bool(readonly),
            "max_fills": int(max_fills),
            "max_orders": int(max_orders),
            "include_completed_orders": True,
        }
        try:
            st.session_state[state_key] = fetch_ibkr_account_snapshot(input_payload=payload)
        except Exception as exc:
            st.error(str(exc))

    snapshot = st.session_state.get(state_key)
    if not isinstance(snapshot, dict):
        st.info("Use the sidebar to fetch a live IBKR account snapshot.")
        return

    warnings = list(snapshot.get("warnings", []))
    if warnings:
        for warning in warnings:
            st.warning(str(warning))

    st.caption(
        f"Snapshot as of `{snapshot.get('as_of', '')}`"
        + f" | Account `{snapshot.get('account_id') or 'auto'}`"
        + f" | Managed Accounts: {', '.join(snapshot.get('managed_accounts', [])) or 'n/a'}"
    )

    dashboard_tab, positions_tab, trades_tab, transactions_tab, orders_tab, raw_tab = st.tabs(
        ["Dashboard", "Positions", "Trades", "Transactions", "Orders", "Raw"]
    )

    with dashboard_tab:
        _render_dashboard(st, snapshot)

    with positions_tab:
        positions_df = _frame(list(snapshot.get("positions", [])))
        st.subheader("Positions")
        if positions_df.empty:
            st.info("No positions returned.")
        else:
            st.dataframe(positions_df, width="stretch", hide_index=True)

    with trades_tab:
        trades_df = _frame(list(snapshot.get("trades", [])))
        st.subheader("Trades")
        if trades_df.empty:
            st.info("No trade rows returned.")
        else:
            st.dataframe(trades_df, width="stretch", hide_index=True)

    with transactions_tab:
        transactions_df = _frame(list(snapshot.get("transactions", [])))
        st.subheader("Transactions")
        if transactions_df.empty:
            st.info("No transaction rows returned.")
        else:
            st.dataframe(transactions_df, width="stretch", hide_index=True)

    with orders_tab:
        orders_df = _frame(list(snapshot.get("orders", [])))
        st.subheader("Orders")
        if orders_df.empty:
            st.info("No order rows returned.")
        else:
            st.dataframe(orders_df, width="stretch", hide_index=True)

    with raw_tab:
        st.subheader("Raw Snapshot")
        st.json(snapshot, expanded=False)


if __name__ == "__main__":
    main()
