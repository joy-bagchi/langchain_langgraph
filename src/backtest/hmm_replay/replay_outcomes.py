from __future__ import annotations

from typing import Any

import pandas as pd


def _realized_regime_label(*, vix_change: float, rv21_change: float, spy_return: float) -> str:
    if vix_change >= 0.1 or rv21_change > 0.0:
        if vix_change >= 0.2:
            return "HIGH_VOL_RISK_OFF"
        return "VOL_EXPANSION_TRANSITION"
    if abs(spy_return) < 0.004:
        return "MID_VOL_CHOP"
    return "STABLE_LOW_VOL_TREND"


def compute_outcome_record(
    frame: pd.DataFrame,
    *,
    as_of_index: int,
    horizons: list[int],
) -> dict[str, Any]:
    as_of_row = frame.iloc[as_of_index]
    result: dict[str, Any] = {
        "as_of_date": str(as_of_row["date"]),
        "spy_close_asof": float(as_of_row["spy_close"]),
        "vix_asof": float(as_of_row["vix"]),
        "vvix_asof": float(as_of_row["vvix"]),
        "rv5_asof": float(as_of_row["realized_vol_5d"]),
        "rv21_asof": float(as_of_row["realized_vol_21d"]),
    }
    for horizon in horizons:
        forward_index = as_of_index + int(horizon)
        if forward_index >= len(frame):
            continue
        future = frame.iloc[forward_index]
        spy_return = (float(future["spy_close"]) / float(as_of_row["spy_close"])) - 1.0
        vix_change = (float(future["vix"]) / float(as_of_row["vix"])) - 1.0
        vvix_change = (float(future["vvix"]) / float(as_of_row["vvix"])) - 1.0
        rv5_change = float(future["realized_vol_5d"]) - float(as_of_row["realized_vol_5d"])
        rv21_change = float(future["realized_vol_21d"]) - float(as_of_row["realized_vol_21d"])
        result.update(
            {
                f"outcome_date_{horizon}d": str(future["date"]),
                f"spy_return_{horizon}d": spy_return,
                f"vix_change_{horizon}d": vix_change,
                f"vvix_change_{horizon}d": vvix_change,
                f"rv5_change_{horizon}d": rv5_change,
                f"rv21_change_{horizon}d": rv21_change,
                f"vix_fell_{horizon}d": bool(vix_change < 0.0),
                f"vix_rose_{horizon}d": bool(vix_change > 0.0),
                f"vix_spike_{horizon}d": bool(vix_change >= 0.10),
                f"rv_expanded_{horizon}d": bool(rv21_change > 0.0),
                f"rv_compressed_{horizon}d": bool(rv21_change < 0.0),
                f"equity_rallied_{horizon}d": bool(spy_return > 0.0),
                f"equity_sold_off_{horizon}d": bool(spy_return < 0.0),
                f"realized_regime_label_{horizon}d": _realized_regime_label(
                    vix_change=vix_change,
                    rv21_change=rv21_change,
                    spy_return=spy_return,
                ),
            }
        )
    return result

