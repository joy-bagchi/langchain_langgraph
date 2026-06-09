"""Simulation-calendar helpers for the strategy game UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class SimulationCalendarPayload:
    quarter_label: str
    start_month: int
    start_day: int
    total_days: int
    end_label: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "quarter_label": self.quarter_label,
            "start_month": self.start_month,
            "start_day": self.start_day,
            "total_days": self.total_days,
            "end_label": self.end_label,
        }


SECONDS_PER_SIMULATED_DAY = 5


def build_simulation_calendar_payload(quarter_label: str) -> SimulationCalendarPayload:
    normalized = quarter_label.strip().upper()
    if normalized == "Q1":
        return SimulationCalendarPayload("Q1", 1, 1, 90, "Mar 31")
    if normalized == "Q2":
        return SimulationCalendarPayload("Q2", 4, 1, 91, "Jun 30")
    if normalized == "Q3":
        return SimulationCalendarPayload("Q3", 7, 1, 92, "Sep 30")
    if normalized == "Q4":
        return SimulationCalendarPayload("Q4", 10, 1, 92, "Dec 31")
    raise ValueError(f"Unsupported quarter label '{quarter_label}'.")


def quarter_duration_seconds(quarter_label: str) -> int:
    payload = build_simulation_calendar_payload(quarter_label)
    return payload.total_days * SECONDS_PER_SIMULATED_DAY


def simulated_date_label(quarter_label: str, *, anchor_epoch: float, now_epoch: float) -> str:
    payload = build_simulation_calendar_payload(quarter_label)
    elapsed_days = max(0, min(payload.total_days - 1, int((now_epoch - anchor_epoch) // SECONDS_PER_SIMULATED_DAY)))
    current_date = date(2025, payload.start_month, payload.start_day) + timedelta(days=elapsed_days)
    month_label = current_date.strftime("%b")
    return f"{month_label} {current_date.day}"
