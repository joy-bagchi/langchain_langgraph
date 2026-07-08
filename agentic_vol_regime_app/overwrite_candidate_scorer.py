"""Portfolio-aware overwrite candidate scorer.

Example:
    python scripts/overwrite_candidate_scorer.py \
      --underlying SPY \
      --spot 740.25 \
      --vix 16.8 \
      --leap-contracts 5 \
      --leap-delta 0.80 \
      --candidate-csv examples/overwrite_candidates_sample.csv \
      --hmm-json outputs/latest_hmm_regime.json \
      --output-dir outputs/overwrite_scorer
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCENARIO_SIGMAS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
REQUIRED_CANDIDATE_COLUMNS: tuple[str, ...] = ("strike", "dte", "bid", "ask", "delta", "iv")


@dataclass(frozen=True)
class ScorerConfig:
    underlying: str
    spot: float
    vix: float
    leap_contracts: int
    leap_delta: float
    upside_drag_penalty: float
    min_premium: float
    max_spread_pct: float
    allow_crash_overwrite: bool


@dataclass(frozen=True)
class HmmContext:
    asof: str | None
    regime_probs: dict[str, float]
    selected_regime: str | None


@dataclass(frozen=True)
class DecisionPolicy:
    recommendation_mode: str
    min_premium: float
    min_distance_sigma: float
    allowed_dte: tuple[int, ...]
    block_new_overwrites: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score short-call overwrite candidates against a LEAP position.")
    parser.add_argument("--underlying", required=True, help="Underlying symbol, for example SPY or QQQ.")
    parser.add_argument("--spot", type=float, required=True, help="Current underlying spot price.")
    parser.add_argument("--vix", type=float, required=True, help="Current VIX level.")
    parser.add_argument("--leap-contracts", type=int, required=True, help="Number of long LEAP contracts.")
    parser.add_argument("--leap-delta", type=float, required=True, help="Delta per LEAP contract.")
    parser.add_argument("--candidate-csv", required=True, help="Path to candidate short-call CSV.")
    parser.add_argument("--hmm-json", help="Optional path to HMM regime probability JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for scored outputs.")
    parser.add_argument("--upside-drag-penalty", type=float, default=0.35, help="Penalty multiplier on max upside drag.")
    parser.add_argument("--min-premium", type=float, default=1.40, help="Minimum premium per contract.")
    parser.add_argument("--max-spread-pct", type=float, default=0.25, help="Maximum allowed bid/ask spread percentage.")
    parser.add_argument(
        "--allow-crash-overwrite",
        action="store_true",
        help="Allow scoring candidates even when crash probability would otherwise block new overwrites.",
    )
    return parser.parse_args(argv)


def compute_daily_sigma(*, spot: float, vix: float) -> tuple[float, float]:
    if spot <= 0:
        raise ValueError("Spot must be positive.")
    if vix <= 0:
        raise ValueError("VIX must be positive.")
    daily_sigma_pct = vix / math.sqrt(252.0) / 100.0
    daily_sigma_points = spot * daily_sigma_pct
    return (daily_sigma_pct, daily_sigma_points)


def load_candidates(candidate_csv: str | Path) -> pd.DataFrame:
    path = Path(candidate_csv)
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing required columns: {', '.join(missing)}")

    working = frame.copy()
    if "mid" not in working.columns:
        working["mid"] = (working["bid"] + working["ask"]) / 2.0
    else:
        working["mid"] = working["mid"].where(working["mid"].notna(), (working["bid"] + working["ask"]) / 2.0)
    numeric_columns = ["strike", "dte", "bid", "ask", "mid", "delta", "iv"]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="raise")
    return working


def load_hmm_context(hmm_json: str | Path | None) -> HmmContext | None:
    if hmm_json is None:
        return None
    path = Path(hmm_json)
    if not path.exists():
        raise FileNotFoundError(f"HMM JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    regime_probs = payload.get("regime_probs", {})
    if not isinstance(regime_probs, dict):
        raise ValueError("HMM JSON must contain a `regime_probs` object.")
    normalized = {str(key): float(value) for key, value in regime_probs.items()}
    return HmmContext(
        asof=payload.get("asof"),
        regime_probs=normalized,
        selected_regime=payload.get("selected_regime"),
    )


def normalize_regime_probs(regime_probs: dict[str, Any]) -> dict[str, float]:
    normalized = {str(key).strip().lower(): float(value) for key, value in dict(regime_probs).items()}
    return {
        "low_vol_trend": float(normalized.get("low_vol_trend", normalized.get("stable_low_vol_trend", 0.0)) or 0.0),
        "mid_vol_chop": float(normalized.get("mid_vol_chop", 0.0) or 0.0),
        "vol_expansion": float(
            normalized.get("vol_expansion", normalized.get("vol_expansion_transition", 0.0)) or 0.0
        ),
        "crash": float(normalized.get("crash", normalized.get("high_vol_risk_off", 0.0)) or 0.0),
    }


def build_decision_policy(
    *, hmm_context: HmmContext | None, base_min_premium: float, allow_crash_overwrite: bool
) -> DecisionPolicy:
    if hmm_context is None:
        return DecisionPolicy("NO_HMM_CONTEXT", base_min_premium, 0.35, (1,), False)

    regime_probs = normalize_regime_probs(hmm_context.regime_probs)
    crash = float(regime_probs.get("crash", 0.0) or 0.0)
    vol_expansion = float(regime_probs.get("vol_expansion", 0.0) or 0.0)
    mid_vol_chop = float(regime_probs.get("mid_vol_chop", 0.0) or 0.0)
    low_vol_trend = float(regime_probs.get("low_vol_trend", 0.0) or 0.0)

    if crash >= 0.15:
        return DecisionPolicy("NO_NEW_OVERWRITE", base_min_premium, 0.35, tuple(), not allow_crash_overwrite)
    if vol_expansion >= 0.55:
        return DecisionPolicy("SELECTIVE_ONLY", base_min_premium * 1.25, 0.50, (1,), False)
    if mid_vol_chop >= 0.50:
        return DecisionPolicy("NORMAL_OVERWRITE", base_min_premium, 0.35, (1, 2), False)
    if low_vol_trend >= 0.50:
        return DecisionPolicy("LIGHT_OVERWRITE", base_min_premium, 0.50, (1,), False)
    if any(regime_probs.values()):
        return DecisionPolicy("UNCERTAIN_SELECTIVE", base_min_premium * 1.15, 0.50, (1,), False)
    return DecisionPolicy("NO_HMM_CONTEXT", base_min_premium, 0.35, (1,), False)


def candidate_generation_anchor(*, spot: float, vix: float) -> float:
    _, daily_sigma_points = compute_daily_sigma(spot=spot, vix=vix)
    return float(spot) + 0.5 * float(daily_sigma_points)


def generate_candidates_from_option_chain(
    option_quotes: list[dict[str, Any]],
    *,
    as_of: str,
    target_strike: float,
    dte_choices: list[int],
    strikes_below_target: int,
    strikes_above_target: int,
) -> pd.DataFrame:
    if not option_quotes:
        raise ValueError("IBKR option chain returned no quotes.")
    if not dte_choices:
        raise ValueError("At least one DTE choice is required.")

    as_of_ts = pd.Timestamp(str(as_of)).tz_localize(None)
    rows: list[dict[str, Any]] = []
    for quote in option_quotes:
        if str(quote.get("right", "")).upper() != "C":
            continue
        expiry_text = str(quote.get("expiry", "")).strip()
        if not expiry_text:
            continue
        expiry_ts = pd.to_datetime(expiry_text, format="%Y%m%d", errors="coerce")
        if pd.isna(expiry_ts):
            continue
        dte = int((expiry_ts.normalize() - as_of_ts.normalize()).days)
        rows.append(
            {
                "symbol": str(quote.get("symbol", "")),
                "expiry": expiry_text,
                "strike": float(quote.get("strike", 0.0) or 0.0),
                "dte": dte,
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "mid": quote.get("mark"),
                "delta": dict(quote.get("greeks", {})).get("delta"),
                "iv": dict(quote.get("greeks", {})).get("implied_vol"),
                "volume": quote.get("volume"),
                "open_interest": quote.get("open_interest"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("IBKR option chain returned no call candidates.")
    for column in ["strike", "dte", "bid", "ask", "mid", "delta", "iv"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["mid"] = frame["mid"].where(frame["mid"].notna(), (frame["bid"] + frame["ask"]) / 2.0)
    frame = frame[frame["dte"].isin([int(item) for item in dte_choices])].copy()
    if frame.empty:
        raise ValueError(f"IBKR option chain did not contain any calls for DTE choices {dte_choices}.")
    frame["target_distance_abs"] = (frame["strike"] - float(target_strike)).abs()
    strikes_sorted = sorted(set(float(value) for value in frame["strike"].dropna().tolist()))
    if not strikes_sorted:
        raise ValueError("IBKR option chain did not contain any usable strike values.")
    anchor_idx = min(range(len(strikes_sorted)), key=lambda idx: abs(strikes_sorted[idx] - float(target_strike)))
    lower_idx = max(0, anchor_idx - max(int(strikes_below_target), 0))
    upper_idx = min(len(strikes_sorted), anchor_idx + max(int(strikes_above_target), 0) + 1)
    selected_strikes = set(strikes_sorted[lower_idx:upper_idx])
    frame = frame[frame["strike"].isin(selected_strikes)].copy()
    if frame.empty:
        raise ValueError("No option candidates remained after target-strike window filtering.")
    return frame.sort_values(["dte", "target_distance_abs", "strike"]).reset_index(drop=True)


def build_scenario_table(
    candidates: pd.DataFrame,
    *,
    spot: float,
    leap_contracts: int,
    leap_delta: float,
    daily_sigma_points: float,
) -> pd.DataFrame:
    scenario_rows: list[dict[str, float]] = []
    short_contracts = leap_contracts
    for candidate in candidates.to_dict(orient="records"):
        strike = float(candidate["strike"])
        premium = float(candidate["mid"])
        dte = float(candidate["dte"])
        for scenario_sigma in SCENARIO_SIGMAS:
            scenario_spot = spot + scenario_sigma * daily_sigma_points
            leap_pnl = leap_contracts * 100.0 * leap_delta * (scenario_spot - spot)
            intrinsic = max(0.0, scenario_spot - strike)
            short_call_pnl = short_contracts * 100.0 * (premium - intrinsic)
            total_pnl = leap_pnl + short_call_pnl
            leap_only_pnl = leap_contracts * 100.0 * leap_delta * (scenario_spot - spot)
            overwrite_drag = leap_only_pnl - total_pnl
            scenario_rows.append(
                {
                    "strike": strike,
                    "dte": dte,
                    "premium": premium,
                    "scenario_sigma": scenario_sigma,
                    "scenario_spot": scenario_spot,
                    "leap_pnl": leap_pnl,
                    "short_call_pnl": short_call_pnl,
                    "total_pnl": total_pnl,
                    "leap_only_pnl": leap_only_pnl,
                    "overwrite_drag": overwrite_drag,
                }
            )
    return pd.DataFrame(scenario_rows)


def apply_decision_rules(
    scored_candidates: pd.DataFrame,
    *,
    decision_policy: DecisionPolicy,
    max_spread_pct: float,
) -> pd.DataFrame:
    decisions: list[str] = []
    reject_reasons: list[str] = []
    for row in scored_candidates.to_dict(orient="records"):
        reasons: list[str] = []
        if decision_policy.block_new_overwrites:
            reasons.append("Crash regime gate blocks new overwrites")
        if int(float(row["dte"])) not in set(int(item) for item in decision_policy.allowed_dte):
            reasons.append(f"DTE not allowed by policy {list(decision_policy.allowed_dte)}")
        if float(row["mid"]) < float(decision_policy.min_premium):
            reasons.append(f"Premium below minimum {decision_policy.min_premium:.2f}")
        if float(row["distance_sigma"]) < float(decision_policy.min_distance_sigma):
            reasons.append(f"Distance sigma below minimum {decision_policy.min_distance_sigma:.2f}")
        if float(row["spread_pct"]) > float(max_spread_pct):
            reasons.append(f"Spread pct above maximum {max_spread_pct:.2f}")
        if float(row["bid"]) <= 0.0:
            reasons.append("Bid must be positive")
        if float(row["mid"]) <= 0.0:
            reasons.append("Mid must be positive")
        if float(row["ask"]) <= float(row["bid"]):
            reasons.append("Ask must be greater than bid")
        decisions.append("ACCEPT" if not reasons else "REJECT")
        reject_reasons.append("; ".join(reasons))

    result = scored_candidates.copy()
    result["decision"] = decisions
    result["reject_reasons"] = reject_reasons
    result["recommendation_mode"] = decision_policy.recommendation_mode
    return result


def score_candidates(
    candidates: pd.DataFrame,
    *,
    config: ScorerConfig,
    hmm_context: HmmContext | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, DecisionPolicy]:
    _, daily_sigma_points = compute_daily_sigma(spot=config.spot, vix=config.vix)
    target_strike = candidate_generation_anchor(spot=config.spot, vix=config.vix)
    scenario_table = build_scenario_table(
        candidates,
        spot=config.spot,
        leap_contracts=config.leap_contracts,
        leap_delta=config.leap_delta,
        daily_sigma_points=daily_sigma_points,
    )
    scenario_metrics = (
        scenario_table[scenario_table["scenario_sigma"].isin([0.0, 0.5, 1.0, 1.5, 2.0])]
        .pivot_table(index=["strike", "dte", "premium"], columns="scenario_sigma", values="total_pnl", aggfunc="first")
        .reset_index()
    )
    scenario_metrics = scenario_metrics.rename(
        columns={
            0.0: "pnl_flat",
            0.5: "pnl_plus_0_5",
            1.0: "pnl_plus_1",
            1.5: "pnl_plus_1_5",
            2.0: "pnl_plus_2",
        }
    )
    max_drag = (
        scenario_table[scenario_table["scenario_sigma"].isin([0.5, 1.0, 1.5, 2.0])]
        .groupby(["strike", "dte", "premium"], as_index=False)["overwrite_drag"]
        .max()
        .rename(columns={"overwrite_drag": "max_upside_drag"})
    )

    scored = candidates.copy()
    scored["underlying"] = config.underlying.upper()
    scored["spot"] = config.spot
    scored["vix"] = config.vix
    scored["daily_sigma_points"] = daily_sigma_points
    scored["target_strike"] = target_strike
    scored["premium"] = scored["mid"]
    scored["distance_from_spot"] = scored["strike"] - config.spot
    scored["distance_sigma"] = scored["distance_from_spot"] / daily_sigma_points
    scored["premium_total"] = scored["mid"] * config.leap_contracts * 100.0
    scored["spread_pct"] = (scored["ask"] - scored["bid"]) / scored["mid"]
    scored = scored.merge(scenario_metrics, on=["strike", "dte", "premium"], how="left")
    scored = scored.merge(max_drag, on=["strike", "dte", "premium"], how="left")
    scored["score"] = scored["premium_total"] - config.upside_drag_penalty * scored["max_upside_drag"]

    decision_policy = build_decision_policy(
        hmm_context=hmm_context,
        base_min_premium=config.min_premium,
        allow_crash_overwrite=config.allow_crash_overwrite,
    )
    scored = apply_decision_rules(scored, decision_policy=decision_policy, max_spread_pct=config.max_spread_pct)
    scored = scored.sort_values(by=["decision", "score"], ascending=[True, False]).reset_index(drop=True)
    return (scored, scenario_table, decision_policy)


def _format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


def _top_rows(frame: pd.DataFrame, *, decision: str, count: int = 5) -> pd.DataFrame:
    filtered = frame[frame["decision"] == decision].copy()
    return filtered.sort_values("score", ascending=False).head(count)


def _most_common_reject_reasons(rejected: pd.DataFrame, *, limit: int = 3) -> list[str]:
    if rejected.empty or "reject_reasons" not in rejected.columns:
        return []
    counts: dict[str, int] = {}
    for raw in rejected["reject_reasons"].astype(str).tolist():
        for part in [item.strip() for item in raw.split(";") if item.strip()]:
            counts[part] = counts.get(part, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [reason for reason, _ in ranked[:limit]]


def build_overwrite_decision_summary(
    *,
    scored_candidates: pd.DataFrame,
    scenario_table: pd.DataFrame,
    metadata: dict[str, Any],
    decision_policy: DecisionPolicy,
    hmm_context: HmmContext | None,
) -> dict[str, Any]:
    accepted = scored_candidates[scored_candidates["decision"] == "ACCEPT"].copy()
    rejected = scored_candidates[scored_candidates["decision"] == "REJECT"].copy()
    accepted = accepted.sort_values("score", ascending=False).reset_index(drop=True)
    rejected = rejected.sort_values("score", ascending=False).reset_index(drop=True)
    best_candidate = accepted.iloc[0].to_dict() if not accepted.empty else None

    target_strike = metadata.get("target_strike")
    if target_strike is None and not scored_candidates.empty and "target_strike" in scored_candidates.columns:
        target_strike = float(pd.to_numeric(scored_candidates["target_strike"], errors="coerce").dropna().iloc[0])
    diagnostic_candidate = None
    if best_candidate is None and not scored_candidates.empty:
        working = scored_candidates.copy()
        if target_strike is not None:
            working["target_distance_abs"] = (pd.to_numeric(working["strike"], errors="coerce") - float(target_strike)).abs()
            working = working.sort_values(["target_distance_abs", "score"], ascending=[True, False])
        else:
            working = working.sort_values("score", ascending=False)
        diagnostic_candidate = working.iloc[0].to_dict()

    mode = str(decision_policy.recommendation_mode).upper()
    if best_candidate is None:
        recommended_action = "NO_OVERWRITE"
    elif mode == "SELECTIVE_ONLY":
        recommended_action = "SELECTIVE_OVERWRITE"
    elif mode == "NORMAL_OVERWRITE":
        recommended_action = "NORMAL_OVERWRITE"
    elif mode == "LIGHT_OVERWRITE":
        recommended_action = "LIGHT_OVERWRITE"
    elif mode == "UNCERTAIN_SELECTIVE":
        recommended_action = "SELECTIVE_OVERWRITE"
    else:
        recommended_action = "SELECTIVE_OVERWRITE"

    regime_probs = normalize_regime_probs(hmm_context.regime_probs) if hmm_context is not None else {}
    top_regime = max(regime_probs, key=regime_probs.get) if regime_probs else None
    top_regime_probability = float(regime_probs.get(top_regime, 0.0)) if top_regime else 0.0
    crash_gate_status = "ACTIVE" if bool(decision_policy.block_new_overwrites) else "OPEN"

    reason_bullets: list[str] = []
    if best_candidate is None:
        if top_regime:
            reason_bullets.append(
                f"HMM context is led by {top_regime.replace('_', ' ')} ({top_regime_probability:.1%})."
            )
        reason_bullets.extend(_most_common_reject_reasons(rejected, limit=3))
        if not reason_bullets:
            reason_bullets.append("No policy-valid candidates were returned.")
        reject_text = " | ".join(rejected.get("reject_reasons", pd.Series(dtype=str)).astype(str).tolist()).lower()
        if decision_policy.block_new_overwrites:
            next_best_action = "Do not open new overwrites under current crash policy."
        elif "dte not allowed by policy" in reject_text:
            next_best_action = (
                "Run again with the policy-allowed DTE chain, or intentionally add this DTE to allowed candidates "
                "if you want to evaluate it."
            )
        elif "premium below minimum" in reject_text:
            next_best_action = "Do not overwrite unless premium improves."
        elif "distance sigma below minimum" in reject_text:
            next_best_action = "Check farther OTM strikes or wait."
        elif "spread pct above maximum" in reject_text:
            next_best_action = "Wait for tighter bid/ask spreads before opening a new overwrite."
        else:
            next_best_action = "Policy is selective right now; wait for better candidates."
        headline = "No overwrite recommended."
    else:
        reason_bullets = [
            f"Distance {float(best_candidate.get('distance_sigma', 0.0)):.2f} sigma is above minimum {decision_policy.min_distance_sigma:.2f}.",
            f"Premium {float(best_candidate.get('mid', 0.0)):.2f} is above minimum {decision_policy.min_premium:.2f}.",
            f"DTE {int(float(best_candidate.get('dte', 0.0)))} is allowed by policy {list(decision_policy.allowed_dte)}.",
            f"Spread {float(best_candidate.get('spread_pct', 0.0)):.2f} is within max {float(metadata.get('max_spread_pct', 0.25)):.2f}.",
            "This candidate has the best portfolio score among accepted candidates.",
            f"Policy mode is {mode}.",
        ]
        next_best_action = "Review the best candidate and scenario PnL before placing any trade manually."
        headline = (
            f"{recommended_action.replace('_', ' ')}: "
            f"{str(metadata.get('underlying', best_candidate.get('underlying', 'SPY'))).upper()} "
            f"{int(float(best_candidate.get('strike', 0.0)))}C "
            f"{int(float(best_candidate.get('dte', 0.0)))}DTE @ {float(best_candidate.get('mid', 0.0)):.2f}"
        )

    return {
        "recommended_action": recommended_action,
        "policy_mode": mode,
        "best_candidate": best_candidate,
        "headline": headline,
        "reason_bullets": reason_bullets,
        "next_best_action": next_best_action,
        "top_regime": top_regime,
        "top_regime_probability": top_regime_probability,
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "crash_gate_status": crash_gate_status,
        "diagnostic_candidate": diagnostic_candidate,
        "selected_regime": hmm_context.selected_regime if hmm_context is not None else None,
        "asof": hmm_context.asof if hmm_context is not None else None,
        "target_strike": float(target_strike) if target_strike is not None else None,
        "has_scenario_rows": bool(not scenario_table.empty),
    }


def _render_markdown_report(
    *,
    config: ScorerConfig,
    hmm_context: HmmContext | None,
    decision_policy: DecisionPolicy,
    scored_candidates: pd.DataFrame,
    scenario_table: pd.DataFrame,
) -> str:
    _, daily_sigma_points = compute_daily_sigma(spot=config.spot, vix=config.vix)
    target_strike = config.spot + 0.5 * daily_sigma_points
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    accepted = _top_rows(scored_candidates, decision="ACCEPT")
    rejected = _top_rows(scored_candidates, decision="REJECT")

    action_summary = "NO OVERWRITE"
    if not accepted.empty:
        best = accepted.iloc[0]
        action_summary = (
            f"{decision_policy.recommendation_mode.replace('_', ' ')}: "
            f"best candidate is {config.underlying.upper()} {int(best['strike'])}C {int(best['dte'])}DTE "
            f"at mid {float(best['mid']):.2f}"
        )

    lines = [
        "# Overwrite Candidate Report",
        "",
        f"- Timestamp: {timestamp}",
        f"- Underlying: {config.underlying.upper()}",
        f"- Spot: {config.spot:.2f}",
        f"- VIX: {config.vix:.2f}",
        f"- LEAP contracts: {config.leap_contracts}",
        f"- LEAP delta: {config.leap_delta:.2f}",
        f"- Recommendation mode: {decision_policy.recommendation_mode}",
        f"- Daily sigma points: {daily_sigma_points:.2f}",
        f"- Heuristic target strike: {target_strike:.2f}",
        f"- Allowed DTE: {list(decision_policy.allowed_dte)}",
        f"- Recommended action: {action_summary}",
        "",
    ]
    if hmm_context is not None:
        lines.extend(["## HMM Context", ""])
        if hmm_context.asof:
            lines.append(f"- As of: {hmm_context.asof}")
        if hmm_context.selected_regime:
            lines.append(f"- Selected regime: {hmm_context.selected_regime}")
        for key, value in sorted(hmm_context.regime_probs.items()):
            lines.append(f"- {key}: {value:.2%}")
        lines.append("")

    if not accepted.empty:
        best = accepted.iloc[0]
        lines.extend(
            [
                "## Recommendation",
                "",
                (
                    f"Best candidate: {config.underlying.upper()} {int(best['strike'])}C {int(best['dte'])}DTE "
                    f"at mid {float(best['mid']):.2f}. Distance is {float(best['distance_sigma']):.2f} sigma. "
                    f"Premium meets threshold, spread is {float(best['spread_pct']):.2f}, "
                    f"and DTE is allowed by policy. Portfolio score is {float(best['score']):.2f}. "
                    f"Recommended only as {decision_policy.recommendation_mode.lower().replace('_', ' ')}."
                ),
                "",
            ]
        )
    else:
        lines.extend(["## Recommendation", "", "No candidate passed the current decision rules.", ""])

    def _append_table_section(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
        lines.extend([title, ""])
        if frame.empty:
            lines.append("No rows.")
            lines.append("")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in frame.iterrows():
            rendered: list[str] = []
            for column in columns:
                value = row[column]
                if column in {"mid", "score", "premium_total", "distance_sigma"}:
                    rendered.append(f"{float(value):.2f}")
                else:
                    rendered.append(str(value))
            lines.append("| " + " | ".join(rendered) + " |")
        lines.append("")

    _append_table_section(
        "## Top Accepted Candidates",
        accepted,
        ["strike", "dte", "mid", "distance_sigma", "premium_total", "score"],
    )
    _append_table_section(
        "## Top Rejected Candidates",
        rejected,
        ["strike", "dte", "mid", "distance_sigma", "score", "reject_reasons"],
    )

    if not accepted.empty:
        best = accepted.iloc[0]
        best_scenarios = scenario_table[
            (scenario_table["strike"] == float(best["strike"])) & (scenario_table["dte"] == float(best["dte"]))
        ].copy()
        lines.extend(["## Best Candidate Scenario PnL", ""])
        lines.append("| Scenario Sigma | Scenario Spot | LEAP PnL | Short Call PnL | Total PnL | LEAP Only PnL | Overwrite Drag |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in best_scenarios.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{float(row['scenario_sigma']):.1f}",
                        _format_currency(row["scenario_spot"]),
                        _format_currency(row["leap_pnl"]),
                        _format_currency(row["short_call_pnl"]),
                        _format_currency(row["total_pnl"]),
                        _format_currency(row["leap_only_pnl"]),
                        _format_currency(row["overwrite_drag"]),
                    ]
                )
                + " |"
            )
            lines.append("")

    if not accepted.empty:
        lines.extend(["## Why This Candidate Passed", ""])
        best = accepted.iloc[0]
        lines.extend(
            [
                f"- Premium {float(best['mid']):.2f} meets minimum {decision_policy.min_premium:.2f}.",
                f"- Distance {float(best['distance_sigma']):.2f} sigma meets minimum {decision_policy.min_distance_sigma:.2f}.",
                f"- DTE {int(float(best['dte']))} is allowed by policy {list(decision_policy.allowed_dte)}.",
                f"- Spread pct {float(best['spread_pct']):.2f} is within limit {float(config.max_spread_pct):.2f}.",
                "",
            ]
        )

    if not rejected.empty:
        lines.extend(["## Why Others Failed", ""])
        for _, row in rejected.head(5).iterrows():
            lines.append(
                f"- Strike {float(row['strike']):.2f} / {int(float(row['dte']))}DTE rejected because {row['reject_reasons']}."
            )
        lines.append("")

    lines.extend(
        [
            "## Warning",
            "",
            "This report is decision support only. It does not execute trades and does not model gamma, vega, skew, early assignment, or liquidity beyond bid/ask spread.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: str | Path,
    config: ScorerConfig,
    hmm_context: HmmContext | None,
    decision_policy: DecisionPolicy,
    scored_candidates: pd.DataFrame,
    scenario_table: pd.DataFrame,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    scored_path = destination / "overwrite_candidates_scored.csv"
    scenario_path = destination / "overwrite_scenario_pnl.csv"
    report_path = destination / "overwrite_report.md"
    snapshot_path = destination / "overwrite_live_snapshot.json"
    markdown_report = _render_markdown_report(
        config=config,
        hmm_context=hmm_context,
        decision_policy=decision_policy,
        scored_candidates=scored_candidates,
        scenario_table=scenario_table,
    )

    scored_candidates.to_csv(scored_path, index=False)
    scenario_table.to_csv(scenario_path, index=False)
    report_path.write_text(markdown_report, encoding="utf-8")
    accepted = scored_candidates[scored_candidates["decision"] == "ACCEPT"].copy()
    _, daily_sigma_points = compute_daily_sigma(spot=config.spot, vix=config.vix)
    best_candidate = accepted.iloc[0].to_dict() if not accepted.empty else None
    snapshot_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "underlying": config.underlying.upper(),
        "spot": float(config.spot),
        "vix": float(config.vix),
        "daily_sigma_points": float(daily_sigma_points),
        "target_strike": float(candidate_generation_anchor(spot=config.spot, vix=config.vix)),
        "hmm_regime_probs": dict(normalize_regime_probs(hmm_context.regime_probs)) if hmm_context is not None else {},
        "selected_regime": hmm_context.selected_regime if hmm_context is not None else None,
        "recommendation_mode": decision_policy.recommendation_mode,
        "best_candidate": best_candidate,
        "input_parameters": {
            "leap_contracts": int(config.leap_contracts),
            "leap_delta": float(config.leap_delta),
            "upside_drag_penalty": float(config.upside_drag_penalty),
            "min_premium": float(config.min_premium),
            "max_spread_pct": float(config.max_spread_pct),
            "allow_crash_overwrite": bool(config.allow_crash_overwrite),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot_payload, indent=2, default=str), encoding="utf-8")
    return {
        "scored_candidates_csv": str(scored_path),
        "scenario_pnl_csv": str(scenario_path),
        "report_md": str(report_path),
        "live_snapshot_json": str(snapshot_path),
        "markdown_report": markdown_report,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ScorerConfig(
        underlying=args.underlying,
        spot=args.spot,
        vix=args.vix,
        leap_contracts=args.leap_contracts,
        leap_delta=args.leap_delta,
        upside_drag_penalty=args.upside_drag_penalty,
        min_premium=args.min_premium,
        max_spread_pct=args.max_spread_pct,
        allow_crash_overwrite=args.allow_crash_overwrite,
    )
    candidates = load_candidates(args.candidate_csv)
    hmm_context = load_hmm_context(args.hmm_json)
    scored_candidates, scenario_table, decision_policy = score_candidates(
        candidates,
        config=config,
        hmm_context=hmm_context,
    )
    outputs = write_outputs(
        output_dir=args.output_dir,
        config=config,
        hmm_context=hmm_context,
        decision_policy=decision_policy,
        scored_candidates=scored_candidates,
        scenario_table=scenario_table,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
