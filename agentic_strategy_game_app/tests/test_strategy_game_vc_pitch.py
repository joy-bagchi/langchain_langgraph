from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import agentic_strategy_game_app.app_runtime as app_runtime
from agentic_strategy_game_app.vc_pitch import (
    append_player_pitch_message,
    apply_vc_agent_response,
    build_vc_agent_input_payload,
    create_vc_pitch_session,
    extract_vc_agent_output_payload,
    parse_vc_agent_response,
)
from agentic_strategy_game_app.scenarios import build_b2b_saas_ai_disruption_scenario


def test_vc_pitch_parser_handles_questioning_response() -> None:
    response = parse_vc_agent_response(
        """
        {
          "mode": "questioning",
          "summary": "I need more evidence before underwriting the round.",
          "diligence_focus": ["growth lever", "revenue projection"],
          "followup_questions": ["What specific growth lever gets you from 5M to 12M ARR?"],
          "tentative_signal": "cautious_interest",
          "decision": null
        }
        """
    )

    assert response.mode == "questioning"
    assert response.decision is None
    assert response.followup_questions


def test_vc_pitch_parser_ignores_trailing_text_after_json() -> None:
    response = parse_vc_agent_response(
        """
        {
          "mode": "questioning",
          "summary": "I need more evidence before underwriting the round.",
          "diligence_focus": ["growth lever", "revenue projection"],
          "followup_questions": ["What specific growth lever gets you from 5M to 12M ARR?"],
          "tentative_signal": "cautious_interest",
          "decision": null
        }
        Additional commentary that should be ignored.
        """
    )

    assert response.mode == "questioning"
    assert response.followup_questions[0].startswith("What specific growth lever")


def test_vc_pitch_parser_synthesizes_summary_when_missing() -> None:
    response = parse_vc_agent_response(
        """
        {
          "mode": "questioning",
          "diligence_focus": ["go to market"],
          "followup_questions": ["What is your payback period by channel?"],
          "tentative_signal": "cautious_interest",
          "decision": null
        }
        """
    )

    assert response.summary == "The investor needs more evidence before making a decision."


def test_vc_pitch_parser_uses_diligence_focus_when_summary_missing() -> None:
    response = parse_vc_agent_response(
        """
        {
          "mode": "questioning",
          "diligence_focus": ["growth levers", "revenue quality", "sales efficiency"],
          "followup_questions": [],
          "tentative_signal": "cautious_interest",
          "decision": null
        }
        """
    )

    assert response.summary == "The investor wants deeper diligence on: growth levers, revenue quality, sales efficiency."


def test_vc_pitch_session_updates_with_agent_response() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    session = create_vc_pitch_session(
        world=world,
        actor_id="ai_native_startup",
        capital_requested=2_500_000.0,
        equity_offered=0.12,
        strategy_summary="We will use capital to accelerate enterprise GTM and product readiness.",
    )
    session = append_player_pitch_message(session, "We are growing quickly and want to scale sales.")
    response = parse_vc_agent_response(
        """
        {
          "mode": "decision",
          "summary": "I will invest, but only on tighter terms.",
          "diligence_focus": ["pipeline quality"],
          "followup_questions": [],
          "tentative_signal": "warm",
          "decision": {
            "outcome": "counter_offer",
            "amount_offered": 2000000,
            "equity_requested": 0.18,
            "terms": ["1x non-participating preference", "monthly investor update"],
            "rationale": "The GTM plan is promising, but execution risk is still high."
          }
        }
        """
    )
    updated = apply_vc_agent_response(session, response)

    assert updated.status == "decision_made"
    assert updated.round_number == 1
    assert updated.latest_agent_response is not None
    assert updated.latest_agent_response.decision is not None
    assert updated.latest_agent_response.decision.outcome == "counter_offer"
    assert updated.transcript[-1].speaker == "vc_agent"


def test_vc_pitch_payload_builder_uses_transcript_and_market_state() -> None:
    _, world = build_b2b_saas_ai_disruption_scenario()
    session = create_vc_pitch_session(
        world=world,
        actor_id="incumbent_platform",
        capital_requested=5_000_000.0,
        equity_offered=0.08,
        strategy_summary="Use capital to accelerate AI copilots and enterprise expansion.",
    )
    session = append_player_pitch_message(session, "We already have paying pilots and want to scale GTM.")
    payload = build_vc_agent_input_payload(session, max_rounds=4)

    assert payload["company_name"] == "Incumbent Platform"
    assert payload["max_rounds"] == 4
    assert "paying pilots" in str(payload["transcript_json"])
    assert "market_size" in str(payload["market_snapshot_json"])


def test_extract_vc_agent_output_payload_prefers_named_outputs() -> None:
    raw = '{"mode":"questioning","summary":"Need more proof.","diligence_focus":[],"followup_questions":["Show me conversion data."],"tentative_signal":"cold","decision":null}'
    result = {
        "named_outputs": {"vc_agent_response": raw},
        "response": "fallback",
    }

    assert extract_vc_agent_output_payload(result) == raw


def test_run_vc_investor_agent_uses_harness_runner(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeObservability:
        def trace_span(self, *args, **kwargs):
            return nullcontext()

    class _FakeServices:
        observability = _FakeObservability()

    def fake_run_agent_workflow(agent_path, input_payload, **kwargs):
        captured["agent_path"] = str(agent_path)
        captured["input_payload"] = dict(input_payload)
        captured["services"] = kwargs.get("services")
        return {"status": "completed", "named_outputs": {"vc_agent_response": "{}"}}

    def fake_build_model_callable(config):
        captured["llm_provider"] = config.provider
        captured["llm_model"] = config.model

        def _model_callable(prompt_text, step, state):
            return "{}"

        return _model_callable

    def fake_build_platform_services(**kwargs):
        captured["model_callable"] = kwargs.get("model_callable")
        return _FakeServices()

    monkeypatch.setattr(app_runtime, "run_agent_workflow", fake_run_agent_workflow)
    monkeypatch.setattr(app_runtime, "build_model_callable", fake_build_model_callable)
    monkeypatch.setattr(app_runtime, "build_platform_services", fake_build_platform_services)

    result = app_runtime.run_vc_investor_agent(
        input_payload={"company_name": "AI-Native Startup", "capital_requested": 2500000},
        agent_path=Path("agentic_strategy_game_app/configs/agents/vc_investor_agent.yaml"),
        storage_root=tmp_path / ".workflow_memory",
    )

    assert result["status"] == "completed"
    assert "vc_investor_agent.yaml" in str(captured["agent_path"])
    assert captured["input_payload"]["capital_requested"] == 2500000
    assert callable(captured["model_callable"])
    assert captured["llm_provider"] == "openai"
    assert captured["llm_model"] == "gpt-4o-mini"
    assert captured["services"] is not None
