"""IV cube shaping and Plotly visualization."""

from __future__ import annotations

import pandas as pd

from vol_surface_publisher.contracts import validate_option_chain_frame

_CALL_COLORS = ("#38bdf8", "#0ea5e9", "#2563eb", "#1d4ed8", "#0891b2", "#0369a1", "#7dd3fc")
_PUT_COLORS = ("#fb7185", "#f43f5e", "#e11d48", "#be123c", "#ec4899", "#db2777", "#fda4af")
_LINE_DASHES = ("solid", "dot", "dash", "dashdot")


def _surface_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a display-ready IV grid from sparse live option observations.

    IBKR can omit individual greeks for otherwise valid contracts. Filling the
    interior grid only for visualization avoids holes in Plotly's mesh while
    leaving the source observations and downloaded table unchanged.
    """
    grid = frame.pivot_table(index="dte", columns="strike", values="implied_vol", aggfunc="mean")
    return grid.sort_index().sort_index(axis=1).interpolate(axis=1, limit_area="inside").interpolate(axis=0, limit_area="inside")


def build_iv_cube(frame: pd.DataFrame, *, right: str | None = None) -> pd.DataFrame:
    """Return the long-form strike × DTE × observation-date IV cube."""
    validate_option_chain_frame(frame)
    cube = frame.copy()
    if right:
        cube = cube[cube["right"] == right.upper()].copy()
    if cube.empty:
        raise ValueError("No IV observations remain after the option-right filter.")
    return cube.sort_values(["observation_date", "dte", "strike", "right"]).reset_index(drop=True)


def select_monthly_atm(frame: pd.DataFrame, *, monthly_offset: int = 0) -> pd.DataFrame:
    """Select each session's monthly expiry at ``monthly_offset`` and its ATM strike.

    A standard monthly expiry is the third Friday. The selected monthly contract
    ``monthly_offset=0`` is the front monthly contract; 1 through 5 select the
    next five monthly contracts. Contracts can roll across dates, so a series
    is defined by monthly maturity position rather than one fixed expiry.
    """
    validate_option_chain_frame(frame)
    if not 0 <= monthly_offset < 6:
        raise ValueError("monthly_offset must be between 0 (front month) and 5.")
    selected: list[pd.DataFrame] = []
    for _, session in frame.groupby(["observation_date", "symbol"], sort=True):
        dates = pd.to_datetime(session["expiry"])
        monthly = session[(dates.dt.weekday == 4) & dates.dt.day.between(15, 21) & (session["dte"] >= 0)].copy()
        if monthly.empty or monthly["underlying_price"].dropna().empty:
            continue
        expiries = monthly.sort_values(["dte", "expiry"])["expiry"].drop_duplicates().tolist()
        if len(expiries) <= monthly_offset:
            continue
        expiry = expiries[monthly_offset]
        expiry_rows = monthly[monthly["expiry"] == expiry].copy()
        underlying = float(expiry_rows["underlying_price"].dropna().iloc[0])
        atm_strike = min(expiry_rows["strike"].unique(), key=lambda value: (abs(float(value) - underlying), float(value)))
        result = expiry_rows[expiry_rows["strike"] == atm_strike].copy()
        result["monthly_expiry"] = expiry
        result["monthly_offset"] = monthly_offset
        result["atm_strike"] = float(atm_strike)
        selected.append(result)
    if not selected:
        raise ValueError("No monthly-expiry ATM observations are available for the selected dates.")
    return pd.concat(selected, ignore_index=True).sort_values(["observation_date", "right"]).reset_index(drop=True)


def select_front_monthly_atm(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible shortcut for the front-monthly ATM selection."""
    return select_monthly_atm(frame, monthly_offset=0)


def create_iv_cube_figure(frame: pd.DataFrame, *, right: str | None = None):
    """Create an interactive 3D view for two or more observation dates."""
    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize the IV cube: `pip install plotly`.") from exc
    cube = build_iv_cube(frame, right=right)
    if cube["observation_date"].nunique() < 2:
        raise ValueError("A 3D IV cube needs at least two observation dates; use the session-surface view instead.")
    figure = px.scatter_3d(cube, x="strike", y="dte", z="observation_date", color="implied_vol", symbol="right",
                           hover_data={"expiry": True, "mark": ":.2f", "underlying_price": ":.2f", "implied_vol": ":.1%"},
                           color_continuous_scale="Viridis",
                           labels={"dte": "Days to expiry", "observation_date": "Observation date", "implied_vol": "Implied vol"},
                           title="Historical implied-volatility cube")
    figure.update_traces(marker={"size": 4, "opacity": 0.85})
    figure.update_layout(coloraxis_colorbar={"tickformat": ".1%", "title": "IV"}, margin={"l": 0, "r": 0, "t": 55, "b": 0})
    figure.update_scenes(xaxis_title="Strike", yaxis_title="Days to expiry", zaxis_title="Observation date")
    return figure


def create_iv_session_figure(frame: pd.DataFrame, *, right: str | None = None):
    """Create a 3D EOD surface: strike × DTE horizontally, IV vertically."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize the IV surface: `pip install plotly`.") from exc
    surface = build_iv_cube(frame, right=right)
    if surface["observation_date"].nunique() != 1:
        raise ValueError("The session-surface view requires exactly one observation date.")
    observed = str(surface["observation_date"].iloc[0])
    symbol = str(surface["symbol"].iloc[0])
    figure = go.Figure()
    for index, option_right in enumerate(sorted(surface["right"].unique())):
        subset = surface[surface["right"] == option_right]
        grid = _surface_grid(subset)
        figure.add_trace(go.Surface(
            x=grid.columns.to_list(),
            y=grid.index.to_list(),
            z=grid.to_numpy(),
            name="Calls" if option_right == "C" else "Puts",
            colorscale=[[0, _CALL_COLORS[0] if option_right == "C" else _PUT_COLORS[0]],
                        [1, _CALL_COLORS[0] if option_right == "C" else _PUT_COLORS[0]]],
            opacity=0.92 if option_right == "C" else 0.62,
            showscale=False,
            showlegend=True,
            hovertemplate="Strike: $%{x:.2f}<br>DTE: %{y}<br>IV: %{z:.1%}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        title=f"{symbol} implied-volatility surface — {observed}",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis": {"title": "Strike", "tickprefix": "$"},
            "yaxis": {"title": "Days to expiry"},
            "zaxis": {"title": "Implied volatility", "tickformat": ".1%"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.8},
            "camera": {"eye": {"x": 1.65, "y": -1.55, "z": 1.05}},
        },
        legend={"title": "Option right"},
    )
    return figure


def create_iv_surface_stack_figure(frame: pd.DataFrame, *, right: str | None = None):
    """Overlay selected EOD surfaces, letting IV height show their changes."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize the IV surfaces: `pip install plotly`.") from exc
    surfaces = build_iv_cube(frame, right=right)
    dates = sorted(surfaces["observation_date"].unique())
    if len(dates) < 2:
        raise ValueError("The stacked-surface view requires at least two observation dates.")
    figure = go.Figure()
    for date_index, observed in enumerate(dates):
        for option_right in sorted(surfaces["right"].unique()):
            subset = surfaces[(surfaces["observation_date"] == observed) & (surfaces["right"] == option_right)]
            if subset.empty:
                continue
            grid = _surface_grid(subset)
            side = "Calls" if option_right == "C" else "Puts"
            palette = _CALL_COLORS if option_right == "C" else _PUT_COLORS
            color = palette[date_index % len(palette)]
            figure.add_trace(go.Surface(
                x=grid.columns.to_list(), y=grid.index.to_list(), z=grid.to_numpy(),
                name=f"{observed} — {side}",
                colorscale=[[0, color], [1, color]],
                opacity=0.76 if option_right == "C" else 0.42,
                showscale=False,
                showlegend=True,
                hovertemplate=(f"Date: {observed}<br>Strike: $%{{x:.2f}}<br>DTE: %{{y}}<br>"
                               f"IV: %{{z:.1%}}<extra>{side}</extra>"),
            ))
    figure.update_layout(
        title="Stacked implied-volatility surfaces",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "xaxis": {"title": "Strike", "tickprefix": "$"},
            "yaxis": {"title": "Days to expiry"},
            "zaxis": {"title": "Implied volatility", "tickformat": ".1%"},
            "aspectmode": "manual", "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.8},
            "camera": {"eye": {"x": 1.65, "y": -1.55, "z": 1.05}},
        },
        legend={"title": "Session and right"},
    )
    return figure


def create_iv_curve_figure(
    frame: pd.DataFrame, *, curve_axis: str, fixed_value: float | int, right: str | None = None
):
    """Plot IV vertically against strike or DTE at one selected cross-section."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize the IV curves: `pip install plotly`.") from exc
    cube = build_iv_cube(frame, right=right)
    if curve_axis == "strike":
        subset = cube[cube["dte"] == int(fixed_value)].copy()
        x_column, title = "strike", f"IV by strike at {int(fixed_value)} DTE"
    elif curve_axis == "dte":
        subset = cube[cube["strike"] == float(fixed_value)].copy()
        x_column, title = "dte", f"IV by DTE at ${float(fixed_value):.2f} strike"
    else:
        raise ValueError("curve_axis must be 'strike' or 'dte'.")
    if subset.empty:
        raise ValueError("No IV observations match the selected curve cross-section.")
    figure = go.Figure()
    date_indexes = {observed: index for index, observed in enumerate(sorted(subset["observation_date"].unique()))}
    for (observed, option_right), group in subset.groupby(["observation_date", "right"], sort=True):
        side = "Calls" if option_right == "C" else "Puts"
        color = _CALL_COLORS[date_indexes[observed] % len(_CALL_COLORS)] if option_right == "C" else _PUT_COLORS[date_indexes[observed] % len(_PUT_COLORS)]
        figure.add_trace(go.Scatter(
            x=group.sort_values(x_column)[x_column], y=group.sort_values(x_column)["implied_vol"],
            mode="lines+markers", name=f"{observed} — {side}",
            line={"color": color, "dash": _LINE_DASHES[date_indexes[observed] % len(_LINE_DASHES)]},
            marker={"color": color},
            hovertemplate=("Strike: $%{x:.2f}<br>IV: %{y:.1%}" if x_column == "strike"
                           else "DTE: %{x}<br>IV: %{y:.1%}") + "<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(title=title, margin={"l": 0, "r": 0, "t": 55, "b": 0}, legend={"title": "Session and right"})
    figure.update_yaxes(title="Implied volatility", tickformat=".1%", gridcolor="rgba(128,128,128,0.2)")
    figure.update_xaxes(title="Strike" if x_column == "strike" else "Days to expiry", tickprefix="$" if x_column == "strike" else None, gridcolor="rgba(128,128,128,0.2)")
    return figure


def create_atm_monthly_history_figure(frame: pd.DataFrame, *, monthly_offset: int = 0):
    """Plot ATM call and put IV for one of the first six monthly maturities."""
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Install Plotly to visualize ATM IV history: `pip install plotly`.") from exc
    history = select_monthly_atm(frame, monthly_offset=monthly_offset)
    if history["observation_date"].nunique() < 2:
        raise ValueError("ATM monthly IV history requires at least two observation dates.")
    figure = go.Figure()
    for option_right in ("C", "P"):
        group = history[history["right"] == option_right].sort_values("observation_date")
        if group.empty:
            continue
        side = "ATM calls" if option_right == "C" else "ATM puts"
        color = _CALL_COLORS[0] if option_right == "C" else _PUT_COLORS[0]
        figure.add_trace(go.Scatter(
            x=group["observation_date"], y=group["implied_vol"], mode="lines+markers", name=side,
            line={"color": color, "width": 3}, marker={"color": color, "size": 8},
            customdata=group[["monthly_expiry", "atm_strike", "dte", "underlying_price"]],
            hovertemplate=("Date: %{x}<br>IV: %{y:.1%}<br>Monthly expiry: %{customdata[0]}<br>"
                           "ATM strike: $%{customdata[1]:.2f}<br>DTE: %{customdata[2]}<br>"
                           "Underlying: $%{customdata[3]:.2f}<extra>%{fullData.name}</extra>"),
        ))
    symbol = str(history["symbol"].iloc[0])
    figure.update_layout(
        title=f"{symbol} {('front monthly' if monthly_offset == 0 else f'{monthly_offset} month' + ('s' if monthly_offset != 1 else '') + ' out')} ATM implied volatility",
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"title": "Option right"},
    )
    figure.update_yaxes(title="Implied volatility", tickformat=".1%", gridcolor="rgba(128,128,128,0.2)")
    figure.update_xaxes(title="Observation date", gridcolor="rgba(128,128,128,0.2)")
    return figure
