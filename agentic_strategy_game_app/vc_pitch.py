"""VC pitch session state and response parsing for the strategy game app."""

from __future__ import annotations

import json
import re
from uuid import uuid4

from agentic_strategy_game_app.contracts import (
    VCPitchAgentResponse,
    VCPitchDecision,
    VCPitchSessionState,
    VCPitchTurn,
    WorldState,
)


JSON_FENCE_PATTERN = re.compile(r"```json\s*(?P<body>\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def create_vc_pitch_session(
    *,
    world: WorldState,
    actor_id: str,
    capital_requested: float,
    equity_offered: float,
    strategy_summary: str,
) -> VCPitchSessionState:
    actor = world.companies.get(actor_id)
    if actor is None:
        raise ValueError(f"Actor '{actor_id}' is not a controllable company in this scenario.")
    return VCPitchSessionState(
        session_id=str(uuid4()),
        actor_id=actor_id,
        company_name=actor.name,
        capital_requested=capital_requested,
        equity_offered=equity_offered,
        strategy_summary=strategy_summary,
        metadata={
            "company_snapshot": actor.to_dict(),
            "market_snapshot": world.market_forces.to_dict(),
        },
    )


def append_player_pitch_message(session: VCPitchSessionState, content: str) -> VCPitchSessionState:
    updated = VCPitchSessionState.from_dict(session.to_dict())
    updated.transcript.append(
        VCPitchTurn(
            speaker="player",
            content=content,
            round_number=updated.round_number,
        )
    )
    return updated


def build_vc_agent_input_payload(
    session: VCPitchSessionState,
    *,
    max_rounds: int = 3,
) -> dict[str, object]:
    company_snapshot = dict(session.metadata.get("company_snapshot", {}))
    market_snapshot = dict(session.metadata.get("market_snapshot", {}))
    transcript = [turn.to_dict() for turn in session.transcript]
    return {
        "company_name": session.company_name,
        "actor_id": session.actor_id,
        "capital_requested": session.capital_requested,
        "equity_offered": session.equity_offered,
        "strategy_summary": session.strategy_summary,
        "round_number": session.round_number,
        "max_rounds": max_rounds,
        "company_snapshot_json": json.dumps(company_snapshot, indent=2, sort_keys=True),
        "market_snapshot_json": json.dumps(market_snapshot, indent=2, sort_keys=True),
        "transcript_json": json.dumps(transcript, indent=2, sort_keys=True),
    }


def _extract_json_payload(raw_text: str) -> dict:
    stripped = raw_text.strip()
    if not stripped:
        raise ValueError("VC agent returned an empty response.")
    fence_match = JSON_FENCE_PATTERN.search(stripped)
    if fence_match:
        stripped = fence_match.group("body").strip()
    decoder = json.JSONDecoder()
    if stripped.startswith("{"):
        try:
            payload, _ = decoder.raw_decode(stripped)
            return payload
        except json.JSONDecodeError:
            pass
    first = stripped.find("{")
    if first >= 0:
        candidate = stripped[first:]
        try:
            payload, _ = decoder.raw_decode(candidate)
            return payload
        except json.JSONDecodeError:
            pass
    raise ValueError("VC agent did not return a parseable JSON object.")


def parse_vc_agent_response(raw_text: str) -> VCPitchAgentResponse:
    payload = _extract_json_payload(raw_text)
    decision_payload = payload.get("decision")
    decision = VCPitchDecision.from_dict(dict(decision_payload)) if decision_payload else None
    summary = str(payload.get("summary", "")).strip()
    followup_questions = [str(item).strip() for item in payload.get("followup_questions", []) if str(item).strip()]
    diligence_focus = [str(item).strip() for item in payload.get("diligence_focus", []) if str(item).strip()]
    mode = str(payload.get("mode", "questioning"))
    if not summary:
        if decision is not None and decision.rationale.strip():
            summary = decision.rationale.strip()
        elif mode == "decision" and decision is not None:
            summary = f"The investor returned a {decision.outcome.replace('_', ' ')} decision."
        elif followup_questions:
            summary = "The investor needs more evidence before making a decision."
        elif diligence_focus:
            focus_preview = ", ".join(diligence_focus[:3])
            summary = f"The investor wants deeper diligence on: {focus_preview}."
        elif str(payload.get("tentative_signal", "")).strip():
            signal = str(payload.get("tentative_signal", "undecided")).replace("_", " ").strip()
            summary = f"The investor's current signal is {signal}."
        else:
            compressed = " ".join(str(raw_text).strip().split())
            if compressed:
                summary = compressed[:220] + ("..." if len(compressed) > 220 else "")
            else:
                summary = "The investor responded, but the pitch response needs a cleaner structure."
    return VCPitchAgentResponse(
        mode=mode,
        summary=summary,
        diligence_focus=diligence_focus,
        followup_questions=followup_questions,
        tentative_signal=str(payload.get("tentative_signal", "undecided")),
        decision=decision,
        raw_response=raw_text,
    )


def extract_vc_agent_output_payload(result: dict[str, object]) -> str:
    named_outputs = result.get("named_outputs")
    if isinstance(named_outputs, dict):
        payload = named_outputs.get("vc_agent_response")
        if isinstance(payload, str) and payload.strip():
            return payload
    response = result.get("response")
    if isinstance(response, str) and response.strip():
        return response
    artifact = result.get("artifact")
    if isinstance(artifact, dict):
        payload = artifact.get("payload")
        if isinstance(payload, dict):
            candidate = payload.get("vc_agent_response")
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    raise ValueError("VC investor run did not return a vc_agent_response payload.")


def apply_vc_agent_response(
    session: VCPitchSessionState,
    response: VCPitchAgentResponse,
) -> VCPitchSessionState:
    updated = VCPitchSessionState.from_dict(session.to_dict())
    updated.round_number += 1
    updated.latest_agent_response = response
    agent_message = response.summary
    if response.followup_questions:
        agent_message = response.summary + "\n\nFollow-up questions:\n- " + "\n- ".join(response.followup_questions)
    updated.transcript.append(
        VCPitchTurn(
            speaker="vc_agent",
            content=agent_message,
            round_number=updated.round_number,
        )
    )
    if response.mode == "decision":
        updated.status = "decision_made"
    return updated
