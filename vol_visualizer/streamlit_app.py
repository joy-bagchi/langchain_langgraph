"""Read-only GCS-backed implied-volatility cube explorer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, not as ``vol_visualizer`` module.
# Ensure sibling packages are importable whether launched from the repo root or
# from within this module directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vol_visualizer.cube import create_atm_monthly_history_figure, create_iv_curve_figure, create_iv_session_figure, create_iv_surface_stack_figure
from vol_visualizer.reader import DEFAULT_BUCKET, DEFAULT_PREFIX, load_option_chain_history, load_surface_catalog


st.set_page_config(page_title="IV Cube Explorer", layout="wide")
st.title("Implied-Volatility Cube Explorer")
st.caption("Read-only explorer for immutable SPY option-surface snapshots published to GCS.")

with st.sidebar:
    st.header("Data source")
    bucket = st.text_input("GCS bucket", value=DEFAULT_BUCKET)
    prefix = st.text_input("GCS prefix", value=DEFAULT_PREFIX)
    project = st.text_input("GCP project (optional)", value="marketphysics") or None

try:
    catalog = load_surface_catalog(bucket=bucket, prefix=prefix, project=project)
except Exception as exc:
    st.error(f"Could not load the surface catalog: {exc}")
    st.stop()

if catalog.empty:
    st.info("No published IV surfaces are available yet.")
    st.stop()

catalog["observation_date"] = pd.to_datetime(catalog["observation_date"]).dt.date
available_dates = sorted(catalog["observation_date"].unique())
symbols = sorted({symbol for row in catalog.get("symbols", []) for symbol in (row or [])}) or ["SPY"]

with st.sidebar:
    selected_dates = st.multiselect("Observation dates", available_dates, default=available_dates[-min(10, len(available_dates)):])
    symbol = st.selectbox("Symbol", symbols)
    right_label = st.selectbox("Option right", ("Calls and puts", "Calls", "Puts"))
    view = st.selectbox("View", ("Single 3D surface", "Stacked 3D surfaces", "IV by strike", "IV by DTE", "ATM monthly Call/Put history"))

if not selected_dates:
    st.info("Choose at least one observation date.")
    st.stop()

start_date, end_date = min(selected_dates).isoformat(), max(selected_dates).isoformat()
try:
    frame = load_option_chain_history(bucket=bucket, prefix=prefix, project=project, start_date=start_date, end_date=end_date)
except Exception as exc:
    st.error(f"Could not load the selected IV data: {exc}")
    st.stop()

frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
frame = frame[(frame["symbol"] == symbol) & (frame["observation_date"].isin(selected_dates))].copy()
right = {"Calls": "C", "Puts": "P"}.get(right_label)

if frame.empty:
    st.warning("The selected catalog entries contain no matching IV observations.")
    st.stop()

left, middle, right_metric = st.columns(3)
left.metric("IV observations", f"{len(frame):,}")
middle.metric("Sessions", frame["observation_date"].nunique())
right_metric.metric("Latest session", max(frame["observation_date"]).isoformat())

surface_date = None
dte = None
strike = None
with st.sidebar:
    st.divider()
    st.subheader("View controls")
    if view == "Single 3D surface":
        surface_date = st.selectbox(
            "Surface date",
            sorted(frame["observation_date"].unique()),
            index=len(frame["observation_date"].unique()) - 1,
        )
    elif view == "IV by strike":
        dte = st.selectbox("Fixed DTE", sorted(int(value) for value in frame["dte"].unique()))
    elif view == "IV by DTE":
        strike = st.selectbox("Fixed strike", sorted(float(value) for value in frame["strike"].unique()))

if view == "Single 3D surface":
    st.plotly_chart(create_iv_session_figure(frame[frame["observation_date"] == surface_date], right=right), use_container_width=True)
elif view == "Stacked 3D surfaces":
    if frame["observation_date"].nunique() < 2:
        st.info("Choose at least two observation dates to stack surfaces.")
    else:
        st.plotly_chart(create_iv_surface_stack_figure(frame, right=right), use_container_width=True)
elif view == "IV by strike":
    st.plotly_chart(create_iv_curve_figure(frame, curve_axis="strike", fixed_value=dte, right=right), use_container_width=True)
elif view == "IV by DTE":
    st.plotly_chart(create_iv_curve_figure(frame, curve_axis="dte", fixed_value=strike, right=right), use_container_width=True)
elif frame["observation_date"].nunique() < 2:
    st.info("ATM Call/Put history appears after at least two published observation dates are selected.")
else:
    st.caption("Each date uses its front standard monthly expiry (third Friday) and the strike closest to that session's underlying price.")
    st.plotly_chart(create_atm_monthly_history_figure(frame), use_container_width=True)
with st.expander("Loaded observations"):
    st.dataframe(frame.sort_values(["observation_date", "dte", "strike", "right"]), use_container_width=True)
