"""Stable tabular contract for dated option implied-volatility observations."""

from __future__ import annotations

from typing import Any

import pandas as pd

OPTION_CHAIN_SCHEMA_VERSION = "option_chain_iv.v1"
MANIFEST_SCHEMA_VERSION = "option_chain_iv_catalog.v1"
REQUIRED_COLUMNS = ("observation_time", "observation_date", "symbol", "expiry", "strike", "dte", "right", "implied_vol")


def option_chain_frame(snapshot: dict[str, Any]) -> pd.DataFrame:
    """Extract valid IV quotes from one normalized IBKR option-chain snapshot."""
    observed_at = pd.Timestamp(snapshot["as_of"], tz="UTC")
    chain = dict(snapshot.get("option_chain", {}))
    symbol = str(chain.get("underlying_symbol") or next(iter(snapshot.get("symbols", {})), "")).upper()
    rows: list[dict[str, Any]] = []
    for quote in chain.get("option_quotes", []):
        try:
            expiry_day = pd.Timestamp(str(quote["expiry"])).date()
            strike, iv = float(quote["strike"]), float(dict(quote.get("greeks") or {})["implied_vol"])
        except (KeyError, TypeError, ValueError):
            continue
        dte = (expiry_day - observed_at.date()).days
        if iv <= 0 or dte < 0:
            continue
        rows.append({"observation_time": observed_at.isoformat(), "observation_date": observed_at.date().isoformat(),
                     "symbol": symbol, "expiry": expiry_day.isoformat(), "strike": strike, "dte": dte,
                     "right": str(quote.get("right", "")).upper(), "implied_vol": iv, "bid": quote.get("bid"),
                     "ask": quote.get("ask"), "mark": quote.get("mark"), "underlying_price": chain.get("underlying_price"),
                     "source": str(snapshot.get("source", "IBKR"))})
    return pd.DataFrame(rows, columns=[*REQUIRED_COLUMNS, "bid", "ask", "mark", "underlying_price", "source"])


def validate_option_chain_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError("Option-surface frame must contain non-empty IV observations and all required columns.")
    if frame.implied_vol.isna().any() or (frame.implied_vol <= 0).any() or frame.dte.isna().any() or (frame.dte < 0).any():
        raise ValueError("Option-surface implied_vol must be positive and dte must be non-negative.")
