"""Public output shaping for artifacts and audience-aware responses."""

from __future__ import annotations

from typing import Any

from agentic_harness.contracts import ArtifactEnvelope, ResponseEnvelope


def _producer_from_result(result: dict[str, Any]) -> dict[str, Any]:
    agent = dict(result.get("agent", {}))
    return {
        "agent_id": result.get("agent_id") or agent.get("agent_id"),
        "agent_name": result.get("agent_name") or agent.get("name"),
        "agent_role": result.get("agent_role") or agent.get("role"),
        "workflow_id": result.get("workflow_id"),
        "run_id": result.get("run_id"),
    }


def _default_payload(result: dict[str, Any]) -> dict[str, Any]:
    named_outputs = dict(result.get("named_outputs", {}))
    if named_outputs:
        return named_outputs
    step_history = list(result.get("step_history", []))
    if step_history:
        last_entry = step_history[-1]
        return {
            "step_id": last_entry.get("step_id"),
            "output": last_entry.get("output"),
            "status": last_entry.get("status"),
        }
    return {}


def extract_artifact(result: dict[str, Any]) -> ArtifactEnvelope:
    """Convert internal run state into a public machine-readable artifact."""
    pending_human_gate = dict(result.get("pending_human_gate") or {})
    if pending_human_gate:
        gate_artifact = pending_human_gate.get("artifact")
        if isinstance(gate_artifact, dict) and "artifact_type" in gate_artifact:
            return ArtifactEnvelope.from_dict(gate_artifact)

    leaf_artifacts = dict(result.get("leaf_artifacts") or {})
    if leaf_artifacts:
        if len(leaf_artifacts) == 1:
            only_artifact = next(iter(leaf_artifacts.values()))
            if isinstance(only_artifact, dict) and "artifact_type" in only_artifact:
                return ArtifactEnvelope.from_dict(only_artifact)
        return ArtifactEnvelope(
            artifact_type="workflow_artifacts",
            version="1.0",
            producer=_producer_from_result(result),
            payload={"artifacts": leaf_artifacts},
            metadata={"status": result.get("status")},
        )

    named_outputs = dict(result.get("named_outputs", {}))
    artifact_type = "workflow_result"
    payload = _default_payload(result)

    if "search_results" in named_outputs:
        artifact_type = "search_results"
        payload = {
            "query": named_outputs.get("search_query"),
            "results": named_outputs["search_results"].get("results", []),
            "raw": named_outputs["search_results"],
        }
    elif "final_brief" in named_outputs:
        artifact_type = "research_brief"
        payload = {"brief": named_outputs["final_brief"], "named_outputs": named_outputs}
    elif len(named_outputs) == 1:
        artifact_type = next(iter(named_outputs))

    return ArtifactEnvelope(
        artifact_type=artifact_type,
        version="1.0",
        producer=_producer_from_result(result),
        payload=payload,
        metadata={
            "status": result.get("status"),
            "current_step": result.get("current_step"),
        },
    )


def _format_human_search_response(artifact: ArtifactEnvelope) -> str:
    query = artifact.payload.get("query") or "search query"
    results = artifact.payload.get("results", [])
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, item in enumerate(results[:5], start=1):
        title = item.get("title") or item.get("url") or "Untitled result"
        url = item.get("url", "")
        snippet = item.get("content") or item.get("snippet") or item.get("raw_content") or ""
        if snippet:
            lines.append(f"{index}. {title} - {url} - {str(snippet).strip()[:240]}")
        else:
            lines.append(f"{index}. {title} - {url}")
    return "\n".join(lines)


def format_response(
    artifact: ArtifactEnvelope,
    *,
    audience: str = "human",
    response_format: str = "auto",
) -> ResponseEnvelope:
    """Render a public response for a human or another agent."""
    resolved_format = response_format
    if response_format == "auto":
        resolved_format = "json" if audience == "agent" else "text"

    if resolved_format == "json":
        content: Any = artifact.to_dict()
    elif artifact.artifact_type == "search_results":
        content = _format_human_search_response(artifact)
    else:
        content = artifact.payload

    return ResponseEnvelope(
        audience=audience,
        response_format=resolved_format,
        content=content,
        metadata={
            "artifact_type": artifact.artifact_type,
            "artifact_version": artifact.version,
        },
    )


def select_output(
    result: dict[str, Any],
    *,
    output_mode: str = "internal",
    audience: str = "human",
    response_format: str = "auto",
) -> dict[str, Any]:
    """Choose which external view of a run to expose."""
    if output_mode == "internal":
        return result

    artifact = extract_artifact(result)
    if output_mode == "artifact":
        return artifact.to_dict()

    response = format_response(
        artifact,
        audience=audience,
        response_format=response_format,
    )
    return response.to_dict()
