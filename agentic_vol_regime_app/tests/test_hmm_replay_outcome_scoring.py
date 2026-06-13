from __future__ import annotations

from src.backtest.hmm_replay.replay_scoring import score_prediction


def test_scoring_uses_frozen_policy_output() -> None:
    prediction = {
        "as_of_date": "2026-06-10",
        "model_name": "heuristic",
        "top_state": "STABLE_LOW_VOL_TREND",
        "state_probabilities": {
            "STABLE_LOW_VOL_TREND": 0.7,
            "MID_VOL_CHOP": 0.2,
            "VOL_EXPANSION_TRANSITION": 0.08,
            "HIGH_VOL_RISK_OFF": 0.02,
        },
        "transition_probabilities": {"to_higher_vol_1d": 0.1},
        "policy_output": {
            "overwrite_posture": "NO_OVERWRITE",
            "suggested_dte": 3,
            "suggested_delta": 0.25,
            "suggested_strike": 760.0,
        },
    }
    outcome = {
        "realized_regime_label_1d": "VOL_EXPANSION_TRANSITION",
        "vix_rose_1d": True,
        "rv_expanded_1d": True,
        "vix_spike_1d": False,
    }

    scored = score_prediction(prediction, outcome, horizon=1)

    assert scored["score_available"] is True
    assert scored["posture"] == "NO_OVERWRITE"
    assert scored["posture_consistent"] is False
    assert scored["state_match"] is False
