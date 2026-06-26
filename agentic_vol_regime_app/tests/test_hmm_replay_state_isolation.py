from __future__ import annotations

from src.backtest.hmm_replay.replay_predictions import create_replay_context


def test_replay_context_is_isolated_per_as_of_date() -> None:
    context_one = create_replay_context("2026-06-10")
    context_two = create_replay_context("2026-06-11")

    assert context_one.run_id != context_two.run_id
    assert context_one.as_of_date == "2026-06-10"
    assert context_two.as_of_date == "2026-06-11"
    assert context_one.allow_live_data is False
    assert context_one.allow_production_artifact_write is False
    assert context_two.allow_live_data is False
    assert context_two.allow_production_artifact_write is False
