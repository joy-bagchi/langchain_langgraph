"""Context engineering layer for assembling step execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from agentic_harness.contracts import WorkflowDefinition, WorkflowGraphState, WorkflowStep


def to_namespace(value: Any) -> Any:
    """Convert dictionaries into attribute-access namespaces for templating."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


class SafeFormatDict(dict):
    """Preserve unknown template tokens instead of failing hard."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(slots=True)
class ContextSnapshot:
    """Assembled context made available to step execution."""

    step_id: str
    current_task: str
    memory_summary: str
    compacted_history: str
    recent_history: list[dict[str, Any]] = field(default_factory=list)
    working_notes: str = ""
    context_brief: str = ""
    raw_memory_hits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "current_task": self.current_task,
            "memory_summary": self.memory_summary,
            "compacted_history": self.compacted_history,
            "recent_history": list(self.recent_history),
            "working_notes": self.working_notes,
            "context_brief": self.context_brief,
            "raw_memory_hits": list(self.raw_memory_hits),
        }


class ContextManager:
    """Deterministic context packer and compactor for workflow steps."""

    def __init__(
        self,
        *,
        workflow_definition: WorkflowDefinition,
        max_recent_history: int = 3,
        max_memory_hits: int = 3,
    ) -> None:
        self.workflow_definition = workflow_definition
        self.max_recent_history = max_recent_history
        self.max_memory_hits = max_memory_hits

    def _history_window(self, step: WorkflowStep) -> int:
        return int(step.metadata.get("max_history_items", self.max_recent_history))

    def _memory_window(self, step: WorkflowStep) -> int:
        return int(step.metadata.get("max_memory_items", self.max_memory_hits))

    def _compact_history(self, step_history: list[dict[str, Any]], *, history_window: int) -> tuple[str, list[dict[str, Any]]]:
        if not step_history:
            return "", []
        recent_history = step_history[-history_window:]
        older_history = step_history[:-history_window] if len(step_history) > history_window else []
        if not older_history:
            return "", recent_history

        compacted_lines = []
        for item in older_history:
            compacted_lines.append(
                f"{item['step_id']} -> {item['status']} :: {str(item.get('output', ''))[:120]}"
            )
        return "\n".join(compacted_lines), recent_history

    def build_context(self, step: WorkflowStep, state: WorkflowGraphState) -> ContextSnapshot:
        """Assemble the active context packet for the current step."""
        memory_hits = list(state.get("memory_hits", []))[: self._memory_window(step)]
        memory_summary = "\n".join(
            item["record"]["content"] for item in memory_hits
        )
        step_history = list(state.get("step_history", []))
        compacted_history, recent_history = self._compact_history(
            step_history,
            history_window=self._history_window(step),
        )
        working_notes = "\n".join(state.get("working_notes", []))
        current_task = (
            step.prompt
            or step.notes
            or f"Execute workflow step {step.step_id}"
        )

        brief_parts = [
            f"Workflow: {self.workflow_definition.title}",
            f"Step: {step.step_id}",
        ]
        if compacted_history:
            brief_parts.append(f"Compacted history available ({len(step_history) - len(recent_history)} prior steps).")
        if memory_summary:
            brief_parts.append(f"Retrieved memory hits: {len(memory_hits)}.")
        if working_notes:
            brief_parts.append("Working notes available.")

        return ContextSnapshot(
            step_id=step.step_id,
            current_task=current_task,
            memory_summary=memory_summary,
            compacted_history=compacted_history,
            recent_history=recent_history,
            working_notes=working_notes,
            context_brief=" ".join(brief_parts),
            raw_memory_hits=memory_hits,
        )


def render_template(
    template: str | None,
    state: WorkflowGraphState,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Render a prompt template against state plus the active context packet."""
    if not template:
        return ""
    active_context = dict(state.get("active_context", {}))
    context = {
        "input": to_namespace(dict(state.get("input_payload", {}))),
        "steps": to_namespace(
            {
                key: {"output": value}
                for key, value in dict(state.get("step_outputs", {})).items()
            }
        ),
        "outputs": to_namespace(dict(state.get("named_outputs", {}))),
        "memory_hits": to_namespace(active_context.get("raw_memory_hits", state.get("memory_hits", []))),
        "memory_summary": active_context.get(
            "memory_summary",
            "\n".join(item["record"]["content"] for item in state.get("memory_hits", [])),
        ),
        "working_notes": active_context.get(
            "working_notes",
            "\n".join(state.get("working_notes", [])),
        ),
        "current_step": state.get("current_step"),
        "compacted_history": active_context.get("compacted_history", ""),
        "recent_history": to_namespace(active_context.get("recent_history", [])),
        "context_brief": active_context.get("context_brief", ""),
        "current_task": active_context.get("current_task", ""),
        "context": to_namespace(active_context),
    }
    if extra:
        context.update({key: to_namespace(value) for key, value in extra.items()})
    return template.format_map(SafeFormatDict(context))

