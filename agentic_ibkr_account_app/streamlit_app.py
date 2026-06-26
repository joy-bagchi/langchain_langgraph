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

from agentic_ibkr_account_app.app_runtime import fetch_ibkr_account_snapshot, fetch_ibkr_option_chain  # noqa: E402


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


def _option_chain_symbols(snapshot: dict[str, Any]) -> list[str]:
    positions = list(snapshot.get("positions", []))
    symbols: list[str] = []
    for row in positions:
        symbol = str(row.get("symbol", "")).strip().upper()
        sec_type = str(row.get("sec_type", "")).strip().upper()
        if symbol and sec_type in {"STK", "ETF", ""} and symbol not in symbols:
            symbols.append(symbol)
    if "SPY" not in symbols:
        symbols.insert(0, "SPY")
    return symbols or ["SPY"]


def _build_option_chain_view(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    option_chain = dict(snapshot.get("option_chain", {}))
    quotes = list(option_chain.get("option_quotes", []))
    grouped: dict[str, dict[tuple[float, str], dict[str, Any]]] = {}
    for quote in quotes:
        expiry = str(quote.get("expiry", "")).strip()
        strike = float(quote.get("strike", 0.0) or 0.0)
        right = str(quote.get("right", "")).strip().upper()
        grouped.setdefault(expiry, {})[(strike, right)] = quote

    rendered: dict[str, list[dict[str, Any]]] = {}
    for expiry, by_key in grouped.items():
        strikes = sorted({strike for strike, _ in by_key.keys()})
        rows: list[dict[str, Any]] = []
        for strike in strikes:
            call = by_key.get((strike, "C"), {})
            put = by_key.get((strike, "P"), {})
            call_greeks = dict(call.get("greeks", {}))
            put_greeks = dict(put.get("greeks", {}))
            rows.append(
                {
                    "Call Bid": call.get("bid"),
                    "Call Ask": call.get("ask"),
                    "Call Last": call.get("last"),
                    "Call Delta": call_greeks.get("delta"),
                    "Call Gamma": call_greeks.get("gamma"),
                    "Call Theta": call_greeks.get("theta"),
                    "Call IV%": (
                        round(float(call_greeks.get("implied_vol", 0.0)) * 100.0, 2)
                        if call_greeks.get("implied_vol") is not None
                        else None
                    ),
                    "Strike": strike,
                    "Put Bid": put.get("bid"),
                    "Put Ask": put.get("ask"),
                    "Put Last": put.get("last"),
                    "Put Delta": put_greeks.get("delta"),
                    "Put Gamma": put_greeks.get("gamma"),
                    "Put Theta": put_greeks.get("theta"),
                    "Put IV%": (
                        round(float(put_greeks.get("implied_vol", 0.0)) * 100.0, 2)
                        if put_greeks.get("implied_vol") is not None
                        else None
                    ),
                }
            )
        rendered[expiry] = rows
    return rendered


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

    dashboard_tab, positions_tab, trades_tab, transactions_tab, orders_tab, option_chain_tab, raw_tab = st.tabs(
        ["Dashboard", "Positions", "Trades", "Transactions", "Orders", "Option Chain", "Raw"]
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

    with option_chain_tab:
        st.subheader("Option Chain")
        available_symbols = _option_chain_symbols(snapshot)
        oc_col1, oc_col2, oc_col3, oc_col4 = st.columns(4)
        option_symbol = oc_col1.selectbox("Underlying", options=available_symbols, index=0, key="option_chain_symbol")
        option_strike_count = oc_col2.number_input(
            "Total Strikes",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            key="option_chain_strike_count",
        )
        option_expiry_count = oc_col3.number_input(
            "Expiry Count",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            key="option_chain_expiry_count",
        )
        option_market_data_type = oc_col4.number_input(
            "Market Data Type",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            key="option_chain_market_data_type",
        )

        if st.button("Load Option Chain", type="primary", key="load_option_chain_button"):
            chain_payload = {
                "host": host.strip() or "127.0.0.1",
                "port": int(port),
                "client_id": int(client_id) + 1000,
                "readonly": bool(readonly),
                "symbol": str(option_symbol).strip().upper() or "SPY",
                "exchange": "SMART",
                "option_exchange": "SMART",
                "currency": "USD",
                "expiry_count": int(option_expiry_count),
                "strike_count": int(option_strike_count),
                "market_data_type": int(option_market_data_type),
                "min_days_to_expiry": 0,
            }
            try:
                st.session_state["agentic_ibkr_option_chain_snapshot"] = fetch_ibkr_option_chain(
                    input_payload=chain_payload
                )
            except Exception as exc:
                st.error(str(exc))

        option_snapshot = st.session_state.get("agentic_ibkr_option_chain_snapshot")
        if isinstance(option_snapshot, dict):
            option_chain = dict(option_snapshot.get("option_chain", {}))
            underlying_price = option_chain.get("underlying_price")
            st.caption(
                f"Underlying `{option_chain.get('underlying_symbol', '')}`"
                + f" | Price `{_format_metric_value(underlying_price)}`"
                + f" | Expiries loaded: {len(option_chain.get('expirations', []))}"
                + f" | Total strikes requested: {int(option_strike_count)}"
            )
            expiry_views = _build_option_chain_view(option_snapshot)
            if not expiry_views:
                st.info("No option chain rows returned.")
            else:
                expiry_tabs = st.tabs(list(expiry_views.keys()))
                for expiry, expiry_tab in zip(expiry_views.keys(), expiry_tabs):
                    with expiry_tab:
                        expiry_df = _frame(expiry_views[expiry])
                        if expiry_df.empty:
                            st.info("No rows returned for this expiry.")
                        else:
                            st.dataframe(expiry_df, width="stretch", hide_index=True)

    with raw_tab:
        st.subheader("Raw Snapshot")
        st.json(snapshot, expanded=False)


if __name__ == "__main__":
    main()
