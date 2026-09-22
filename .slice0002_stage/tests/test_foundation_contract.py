from pathlib import Path
import math
import tomllib

import pytest


ROOT = Path(__file__).parents[1]


def test_vix_scaled_baseline_defaults_and_history_identity():
    with (ROOT / "configs" / "skew_baseline.example.toml").open("rb") as stream:
        config = tomllib.load(stream)

    assert config["metric_id"] == "spy_iv_skew_vix_scaled"
    assert config["methodology_version"] == "v2"
    assert config["sigma_multiplier"] == 0.5
    assert config["annualization_trading_sessions"] == 252
    assert config["strike_distance_policy"] == "vix_horizon_scaled_half_volatility"
    assert (
        config["inputs"]["volatility_history_manifest_object"]
        == "market-manifold/vol-regime-history/manifests/latest.json"
    )

    contract = (ROOT / "docs" / "baseline-history-contract.md").read_text(
        encoding="utf-8"
    )
    assert "spy_iv_skew_vix_scaled/<sigma_multiplier_id>" in contract
    assert "spy_iv_skew_pm1pct" not in contract
    assert "sqrt(trading_session_DTE / 252)" in contract
    assert "shared one-day reference distance" in contract

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MARKET_PHYSICS_VIX_SOURCE=" in env_example


def test_horizon_scaled_reference_targets():
    spot = 760.0
    vix = 15.87
    sigma_multiplier = 0.5

    distances = {
        dte: spot * sigma_multiplier * (vix / 100) * math.sqrt(dte / 252)
        for dte in (1, 2)
    }

    assert distances[1] == pytest.approx(3.8, abs=0.01)
    assert distances[2] == pytest.approx(5.37, abs=0.01)
    assert spot - distances[1] == pytest.approx(756.2, abs=0.01)
    assert spot + distances[1] == pytest.approx(763.8, abs=0.01)
    assert spot - distances[2] == pytest.approx(754.63, abs=0.01)
    assert spot + distances[2] == pytest.approx(765.37, abs=0.01)
