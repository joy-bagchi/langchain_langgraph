from __future__ import annotations

import math

import pandas as pd

from agentic_vol_regime_app.overwrite_candidate_scorer import (
    HmmContext,
    ScorerConfig,
    DecisionPolicy,
    build_overwrite_decision_summary,
    build_decision_policy,
    build_scenario_table,
    compute_daily_sigma,
    generate_candidates_from_option_chain,
    load_candidates,
    score_candidates,
)


def test_compute_daily_sigma() -> None:
    daily_sigma_pct, daily_sigma_points = compute_daily_sigma(spot=740.25, vix=16.8)
    assert math.isclose(daily_sigma_pct, 16.8 / math.sqrt(252.0) / 100.0)
    assert math.isclose(daily_sigma_points, 740.25 * daily_sigma_pct)


def test_load_candidates_fills_mid(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    candidate_csv.write_text("strike,dte,bid,ask,delta,iv\n743,1,1.46,1.58,0.22,0.161\n", encoding="utf-8")
    loaded = load_candidates(candidate_csv)
    assert loaded.loc[0, "mid"] == 1.52


def test_rejects_premium_below_threshold() -> None:
    candidates = pd.DataFrame(
        [{"strike": 743.0, "dte": 1.0, "bid": 1.00, "ask": 1.20, "mid": 1.10, "delta": 0.22, "iv": 0.16}]
    )
    config = ScorerConfig(
        underlying="SPY",
        spot=740.25,
        vix=16.8,
        leap_contracts=5,
        leap_delta=0.80,
        upside_drag_penalty=0.35,
        min_premium=1.40,
        max_spread_pct=0.25,
        allow_crash_overwrite=False,
    )
    scored, _, _ = score_candidates(candidates, config=config)
    assert scored.loc[0, "decision"] == "REJECT"
    assert "Premium below minimum 1.40" in scored.loc[0, "reject_reasons"]


def test_vol_expansion_raises_threshold_and_distance_requirement() -> None:
    candidates = pd.DataFrame(
        [{"strike": 743.0, "dte": 1.0, "bid": 1.70, "ask": 1.80, "mid": 1.75, "delta": 0.22, "iv": 0.16}]
    )
    config = ScorerConfig(
        underlying="SPY",
        spot=740.25,
        vix=16.8,
        leap_contracts=5,
        leap_delta=0.80,
        upside_drag_penalty=0.35,
        min_premium=1.40,
        max_spread_pct=0.25,
        allow_crash_overwrite=False,
    )
    hmm_context = HmmContext(
        asof="2026-06-25",
        regime_probs={"vol_expansion": 0.55, "mid_vol_chop": 0.20, "low_vol_trend": 0.15, "crash": 0.10},
        selected_regime="vol_expansion",
    )
    scored, _, decision_policy = score_candidates(candidates, config=config, hmm_context=hmm_context)
    assert decision_policy.recommendation_mode == "SELECTIVE_ONLY"
    assert math.isclose(decision_policy.min_premium, 1.75)
    assert math.isclose(decision_policy.min_distance_sigma, 0.50)
    assert scored.loc[0, "decision"] == "REJECT"
    assert "Distance sigma below minimum 0.50" in scored.loc[0, "reject_reasons"]


def test_scenario_pnl_math_for_simple_candidate() -> None:
    candidates = pd.DataFrame(
        [{"strike": 105.0, "dte": 1.0, "bid": 2.0, "ask": 2.0, "mid": 2.0, "delta": 0.2, "iv": 0.2}]
    )
    scenario_table = build_scenario_table(
        candidates,
        spot=100.0,
        leap_contracts=1,
        leap_delta=0.5,
        daily_sigma_points=10.0,
    )
    plus_one = scenario_table[scenario_table["scenario_sigma"] == 1.0].iloc[0]
    assert plus_one["scenario_spot"] == 110.0
    assert plus_one["leap_pnl"] == 500.0
    assert plus_one["short_call_pnl"] == -300.0
    assert plus_one["total_pnl"] == 200.0
    assert plus_one["leap_only_pnl"] == 500.0
    assert plus_one["overwrite_drag"] == 300.0


def test_crash_gate_blocks_new_overwrites() -> None:
    policy = build_decision_policy(
        hmm_context=HmmContext(
            asof="2026-06-25",
            regime_probs={"crash": 0.20, "vol_expansion": 0.10, "mid_vol_chop": 0.30, "low_vol_trend": 0.40},
            selected_regime="crash",
        ),
        base_min_premium=1.40,
        allow_crash_overwrite=False,
    )
    assert policy.recommendation_mode == "NO_NEW_OVERWRITE"
    assert policy.block_new_overwrites is True


def test_uncertain_regime_falls_back_to_selective_mode() -> None:
    policy = build_decision_policy(
        hmm_context=HmmContext(
            asof="2026-06-25",
            regime_probs={"crash": 0.10, "vol_expansion": 0.30, "mid_vol_chop": 0.30, "low_vol_trend": 0.30},
            selected_regime="vol_expansion",
        ),
        base_min_premium=1.40,
        allow_crash_overwrite=False,
    )
    assert policy.recommendation_mode == "UNCERTAIN_SELECTIVE"
    assert policy.allowed_dte == (1,)


def test_generate_candidates_from_option_chain_filters_around_target_strike() -> None:
    option_quotes = []
    for strike in [738.0, 739.0, 740.0, 741.0, 742.0, 743.0, 744.0]:
        option_quotes.append(
            {
                "symbol": f"SPY_{int(strike)}C",
                "expiry": "20260626",
                "strike": strike,
                "right": "C",
                "bid": 1.0,
                "ask": 1.2,
                "mark": 1.1,
                "greeks": {"delta": 0.25, "implied_vol": 0.18},
            }
        )
    frame = generate_candidates_from_option_chain(
        option_quotes,
        as_of="2026-06-25T20:00:00Z",
        target_strike=741.0,
        dte_choices=[1],
        strikes_below_target=1,
        strikes_above_target=2,
    )
    assert sorted(frame["strike"].tolist()) == [740.0, 741.0, 742.0, 743.0]


def test_decision_summary_no_accepted_due_to_dte_mismatch() -> None:
    scored = pd.DataFrame(
        [
            {
                "strike": 743.0,
                "dte": 3.0,
                "mid": 1.6,
                "bid": 1.5,
                "ask": 1.7,
                "distance_sigma": 0.6,
                "spread_pct": 0.1,
                "score": 120.0,
                "decision": "REJECT",
                "reject_reasons": "DTE not allowed by policy [1]",
            }
        ]
    )
    summary = build_overwrite_decision_summary(
        scored_candidates=scored,
        scenario_table=pd.DataFrame(),
        metadata={"underlying": "SPY", "max_spread_pct": 0.25, "target_strike": 743.0},
        decision_policy=DecisionPolicy("SELECTIVE_ONLY", 1.4, 0.5, (1,), False),
        hmm_context=HmmContext(asof="2026-06-25", regime_probs={"vol_expansion": 0.6}, selected_regime="vol_expansion"),
    )
    assert summary["recommended_action"] == "NO_OVERWRITE"
    assert "Run again with the policy-allowed DTE chain" in summary["next_best_action"]


def test_decision_summary_no_accepted_due_to_premium() -> None:
    scored = pd.DataFrame(
        [
            {
                "strike": 743.0,
                "dte": 1.0,
                "mid": 1.1,
                "bid": 1.0,
                "ask": 1.2,
                "distance_sigma": 0.6,
                "spread_pct": 0.15,
                "score": 80.0,
                "decision": "REJECT",
                "reject_reasons": "Premium below minimum 1.40",
            }
        ]
    )
    summary = build_overwrite_decision_summary(
        scored_candidates=scored,
        scenario_table=pd.DataFrame(),
        metadata={"underlying": "SPY", "max_spread_pct": 0.25, "target_strike": 743.0},
        decision_policy=DecisionPolicy("NORMAL_OVERWRITE", 1.4, 0.35, (1, 2), False),
        hmm_context=None,
    )
    assert summary["recommended_action"] == "NO_OVERWRITE"
    assert summary["next_best_action"] == "Do not overwrite unless premium improves."


def test_decision_summary_crash_gate_active() -> None:
    scored = pd.DataFrame(
        [
            {
                "strike": 742.0,
                "dte": 1.0,
                "mid": 1.8,
                "bid": 1.7,
                "ask": 1.9,
                "distance_sigma": 0.6,
                "spread_pct": 0.1,
                "score": 100.0,
                "decision": "REJECT",
                "reject_reasons": "Crash regime gate blocks new overwrites",
            }
        ]
    )
    summary = build_overwrite_decision_summary(
        scored_candidates=scored,
        scenario_table=pd.DataFrame(),
        metadata={"underlying": "SPY", "max_spread_pct": 0.25, "target_strike": 742.0},
        decision_policy=DecisionPolicy("NO_NEW_OVERWRITE", 1.4, 0.35, tuple(), True),
        hmm_context=HmmContext(asof="2026-06-25", regime_probs={"crash": 0.2}, selected_regime="crash"),
    )
    assert summary["recommended_action"] == "NO_OVERWRITE"
    assert summary["crash_gate_status"] == "ACTIVE"
    assert "Do not open new overwrites" in summary["next_best_action"]


def test_decision_summary_accepted_selective_candidate() -> None:
    scored = pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "strike": 743.0,
                "dte": 1.0,
                "mid": 1.7,
                "bid": 1.6,
                "ask": 1.8,
                "distance_sigma": 0.6,
                "spread_pct": 0.1,
                "score": 150.0,
                "decision": "ACCEPT",
                "reject_reasons": "",
            }
        ]
    )
    summary = build_overwrite_decision_summary(
        scored_candidates=scored,
        scenario_table=pd.DataFrame([{"strike": 743.0, "dte": 1.0, "scenario_sigma": 0.0, "total_pnl": 0.0}]),
        metadata={"underlying": "SPY", "max_spread_pct": 0.25, "target_strike": 743.0},
        decision_policy=DecisionPolicy("SELECTIVE_ONLY", 1.4, 0.5, (1,), False),
        hmm_context=HmmContext(asof="2026-06-25", regime_probs={"vol_expansion": 0.6}, selected_regime="vol_expansion"),
    )
    assert summary["recommended_action"] == "SELECTIVE_OVERWRITE"
    assert summary["best_candidate"] is not None


def test_decision_summary_accepted_normal_candidate() -> None:
    scored = pd.DataFrame(
        [
            {
                "underlying": "SPY",
                "strike": 744.0,
                "dte": 2.0,
                "mid": 1.5,
                "bid": 1.45,
                "ask": 1.55,
                "distance_sigma": 0.4,
                "spread_pct": 0.07,
                "score": 140.0,
                "decision": "ACCEPT",
                "reject_reasons": "",
            }
        ]
    )
    summary = build_overwrite_decision_summary(
        scored_candidates=scored,
        scenario_table=pd.DataFrame([{"strike": 744.0, "dte": 2.0, "scenario_sigma": 0.0, "total_pnl": 0.0}]),
        metadata={"underlying": "SPY", "max_spread_pct": 0.25, "target_strike": 744.0},
        decision_policy=DecisionPolicy("NORMAL_OVERWRITE", 1.4, 0.35, (1, 2), False),
        hmm_context=HmmContext(asof="2026-06-25", regime_probs={"mid_vol_chop": 0.6}, selected_regime="mid_vol_chop"),
    )
    assert summary["recommended_action"] == "NORMAL_OVERWRITE"
