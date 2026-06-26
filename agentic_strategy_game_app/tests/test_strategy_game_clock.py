from __future__ import annotations

from agentic_strategy_game_app.simulation_clock import build_simulation_calendar_payload, quarter_duration_seconds


def test_simulation_calendar_payload_maps_quarters() -> None:
    q1 = build_simulation_calendar_payload("Q1")
    q3 = build_simulation_calendar_payload("Q3")

    assert q1.start_month == 1
    assert q1.start_day == 1
    assert q1.total_days == 90
    assert q1.end_label == "Mar 31"
    assert q3.start_month == 7
    assert q3.end_label == "Sep 30"


def test_quarter_duration_seconds_uses_five_second_days() -> None:
    assert quarter_duration_seconds("Q1") == 450
    assert quarter_duration_seconds("Q2") == 455
