from __future__ import annotations

from src.regime.meta_blend import blend_with_geometry_modifier, core_vol_risk_score, score_to_state


def test_core_vol_risk_score_and_state_mapping() -> None:
    score = core_vol_risk_score(
        {
            "STABLE_LOW_VOL_TREND": 0.1,
            "MID_VOL_CHOP": 0.2,
            "VOL_EXPANSION_TRANSITION": 0.5,
            "HIGH_VOL_RISK_OFF": 0.2,
        }
    )
    assert 0.0 <= score <= 1.0
    assert score_to_state(0.24) == "STABLE_LOW_VOL_TREND"
    assert score_to_state(0.40) == "MID_VOL_CHOP"
    assert score_to_state(0.60) == "VOL_EXPANSION_TRANSITION"
    assert score_to_state(0.90) == "HIGH_VOL_RISK_OFF"


def test_meta_blend_respects_weights() -> None:
    result_a = blend_with_geometry_modifier(
        core_vol_state="VOL_EXPANSION_TRANSITION",
        core_vol_confidence=0.7,
        core_vol_risk_score_value=0.67,
        geometry_stress_score=0.20,
        core_vol_weight=0.85,
        geometry_weight=0.15,
    )
    result_b = blend_with_geometry_modifier(
        core_vol_state="VOL_EXPANSION_TRANSITION",
        core_vol_confidence=0.7,
        core_vol_risk_score_value=0.67,
        geometry_stress_score=0.20,
        core_vol_weight=0.65,
        geometry_weight=0.35,
    )
    assert result_a.final_risk_score > result_b.final_risk_score


def test_downgrade_cap_prevents_over_downgrade_without_exception() -> None:
    result = blend_with_geometry_modifier(
        core_vol_state="VOL_EXPANSION_TRANSITION",
        core_vol_confidence=0.8,
        core_vol_risk_score_value=0.67,
        geometry_stress_score=0.01,
        core_vol_weight=0.10,
        geometry_weight=0.90,
    )
    assert result.final_regime in {"MID_VOL_CHOP", "VOL_EXPANSION_TRANSITION"}
    assert result.final_regime != "STABLE_LOW_VOL_TREND"
    assert result.downgrade_cap_applied is True


def test_strict_exception_allows_deep_downgrade() -> None:
    result = blend_with_geometry_modifier(
        core_vol_state="VOL_EXPANSION_TRANSITION",
        core_vol_confidence=0.50,
        core_vol_risk_score_value=0.67,
        geometry_stress_score=0.01,
        core_vol_weight=0.10,
        geometry_weight=0.90,
    )
    assert result.final_regime == "STABLE_LOW_VOL_TREND"
