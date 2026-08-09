"""IV cube shaping and Plotly visualization."""

from __future__ import annotations

import pandas as pd

from vol_surface_publisher.contracts import validate_option_chain_frame


def build_iv_cube(frame: pd.DataFrame, *, right: str | None = None) -> pd.DataFrame:
    """Return the long-form strike × DTE × observation-date IV cube."""
    validate_option_chain_frame(frame)
    cube = frame.copy()
    if right:
        cube = cube[cube["right"] == right.upper()].copy()
    if cube.empty:
        raise ValueError("No IV observations remain after the option-right filter.")
    return cube.sort_values(["observation_date", "dte", "strike", "right"]).reset_index(drop=True)


def create_iv_cube_figure(frame: pd.DataFrame, *, right: str | None = None):
    """Create an interactive standard Plotly 3D scatter of the IV cube."""
    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize the IV cube: `pip install plotly`.") from exc
    cube = build_iv_cube(frame, right=right)
    return px.scatter_3d(cube, x="strike", y="dte", z="observation_date", color="implied_vol", symbol="right",
                         hover_data=["expiry", "mark", "underlying_price"], color_continuous_scale="Viridis",
                         labels={"dte": "DTE", "observation_date": "Observation date", "implied_vol": "Implied vol"},
                         title="Implied-volatility cube")
