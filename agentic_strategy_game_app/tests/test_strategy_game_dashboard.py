from __future__ import annotations

from agentic_strategy_game_app.dashboard import (
    apply_market_force_overrides,
    company_metric_correlation_rows,
    load_scenario_world,
    strategic_pressure_summary,
)


def test_apply_market_force_overrides_returns_cloned_world() -> None:
    _, world = load_scenario_world()

    updated = apply_market_force_overrides(
        world,
        {"regulatory_pressure": 0.61, "technology_shift_intensity": 0.95},
    )

    assert updated.market_forces.regulatory_pressure == 0.61
    assert updated.market_forces.technology_shift_intensity == 0.95
    assert world.market_forces.regulatory_pressure == 0.24
    assert world.market_forces.technology_shift_intensity == 0.88


def test_strategic_pressure_summary_stays_bounded() -> None:
    _, world = load_scenario_world()

    summary = strategic_pressure_summary(world.market_forces)

    assert set(summary) == {
        "disruption_pressure",
        "commercial_headwind",
        "resource_flexibility",
        "regulatory_heat",
    }
    assert all(0.0 <= value <= 1.0 for value in summary.values())


def test_company_metric_correlation_rows_build_square_matrix() -> None:
    _, world = load_scenario_world()

    rows = company_metric_correlation_rows(world)
    metric_names = {row["metric"] for row in rows}

    assert len(rows) == len(metric_names)
    for row in rows:
        for metric_name in metric_names:
            assert metric_name in row
