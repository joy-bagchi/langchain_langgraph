"""Regime labels and helpers."""

from __future__ import annotations


REGIMES = (
    "STABLE_LOW_VOL_TREND",
    "MID_VOL_CHOP",
    "VOL_EXPANSION_TRANSITION",
    "HIGH_VOL_RISK_OFF",
)

REGIME_LABELS = {
    "STABLE_LOW_VOL_TREND": "Stable Low-Vol Trend",
    "MID_VOL_CHOP": "Mid-Vol Chop",
    "VOL_EXPANSION_TRANSITION": "Vol Expansion Transition",
    "HIGH_VOL_RISK_OFF": "High-Vol Risk-Off",
}
