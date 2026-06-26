"""Parser for the structured markdown workflow DSL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from agentic_harness.contracts import (
    BranchRule,
    MemoryWritePolicy,
    WorkflowDefinition,
    WorkflowStep,
)


FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
STEP_HEADER_PATTERN = re.compile(r"^##\s+Step:\s*(.+?)\s*$", re.MULTILINE)
FENCED_BLOCK_PATTERN = re.compile(
    r"```(?P<label>[A-Za-z0-9_-]+)?\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    return normalized.strip("_")


def _extract_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_PATTERN.match(markdown)
    if not match:
        raise ValueError("Workflow markdown must start with YAML frontmatter.")
    payload = yaml.safe_load(match.group(1)) or {}
    return payload, markdown[match.end() :]


def _parse_step_blocks(section_body: str) -> tuple[dict[str, Any], str | None, str]:
    metadata: dict[str, Any] | None = None
    prompt: str | None = None
    consumed_spans: list[tuple[int, int]] = []

    for match in FENCED_BLOCK_PATTERN.finditer(section_body):
        label = (match.group("label") or "").strip().lower()
        body = match.group("body").strip()
        consumed_spans.append(match.span())
        if label == "yaml":
            metadata = yaml.safe_load(body) or {}
        elif label == "prompt":
            prompt = body

    if metadata is None:
        raise ValueError("Each step must include a ```yaml``` metadata block.")

    notes_parts: list[str] = []
    cursor = 0
    for start, end in consumed_spans:
        notes_parts.append(section_body[cursor:start])
        cursor = end
    notes_parts.append(section_body[cursor:])
    notes = "\n".join(part.strip() for part in notes_parts if part.strip()).strip()
    return metadata, prompt, notes


def _parse_step(step_heading: str, section_body: str) -> WorkflowStep:
    metadata, prompt, notes = _parse_step_blocks(section_body)
    step_id = str(metadata.get("id") or _slugify(step_heading))
    branches = [BranchRule.from_dict(item) for item in metadata.get("branches", [])]
    memory = MemoryWritePolicy.from_dict(metadata.get("memory"))
    step_type = str(metadata.get("type", "prompt")).strip()
    title = str(metadata.get("title") or step_heading).strip()

    return WorkflowStep(
        step_id=step_id,
        title=title,
        step_type=step_type,
        output_key=metadata.get("output_key"),
        next_step=metadata.get("next"),
        branches=branches,
        approved_next=metadata.get("approved_next"),
        rejected_next=metadata.get("rejected_next"),
        prompt=prompt,
        executor=metadata.get("executor"),
        max_retries=int(metadata.get("max_retries", 0)),
        memory=memory,
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "id",
                "title",
                "type",
                "output_key",
                "next",
                "branches",
                "approved_next",
                "rejected_next",
                "max_retries",
                "memory",
                "executor",
            }
        },
        notes=notes,
    )


def _validate_workflow(definition: WorkflowDefinition) -> None:
    if definition.entry_step not in definition.steps:
        raise ValueError(
            f"Entry step '{definition.entry_step}' is not defined in the workflow."
        )

    for step in definition.steps.values():
        referenced_steps = [
            value
            for value in [
                step.next_step,
                step.approved_next,
                step.rejected_next,
                *[branch.next_step for branch in step.branches],
            ]
            if value is not None
        ]
        for target in referenced_steps:
            if target not in definition.steps:
                raise ValueError(
                    f"Step '{step.step_id}' references unknown next step '{target}'."
                )

        if step.step_type == "human_review":
            if not step.approved_next or not step.rejected_next:
                raise ValueError(
                    f"Human review step '{step.step_id}' requires "
                    "'approved_next' and 'rejected_next'."
                )


def parse_workflow_markdown(markdown: str, *, workflow_path: str | None = None) -> WorkflowDefinition:
    """Parse markdown text into a typed workflow definition."""
    frontmatter, body = _extract_frontmatter(markdown)
    matches = list(STEP_HEADER_PATTERN.finditer(body))
    if not matches:
        raise ValueError("No workflow steps found. Use '## Step: <name>' headings.")

    steps: dict[str, WorkflowStep] = {}
    for index, match in enumerate(matches):
        section_start = match.end()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        step = _parse_step(match.group(1), body[section_start:section_end].strip())
        if step.step_id in steps:
            raise ValueError(f"Duplicate step id '{step.step_id}' in workflow.")
        steps[step.step_id] = step

    metadata = dict(frontmatter)
    workflow_id = str(metadata.pop("workflow_id", metadata.get("title", "workflow")))
    title = str(metadata.pop("title", workflow_id)).strip()
    entry_step = str(metadata.pop("entry_step"))
    memory_namespace = str(
        metadata.pop("memory_namespace", f"{_slugify(workflow_id)}_memory")
    )
    default_model = metadata.pop("default_model", None)
    description = str(metadata.pop("description", "")).strip()

    definition = WorkflowDefinition(
        workflow_id=workflow_id,
        title=title,
        entry_step=entry_step,
        steps=steps,
        memory_namespace=memory_namespace,
        default_model=default_model,
        description=description,
        workflow_path=workflow_path,
        metadata=metadata,
    )
    _validate_workflow(definition)
    return definition


def load_workflow_definition(path: str | Path) -> WorkflowDefinition:
    """Load and parse a workflow markdown file from disk."""
    workflow_path = Path(path).resolve()
    return parse_workflow_markdown(
        workflow_path.read_text(encoding="utf-8"),
        workflow_path=str(workflow_path),
    )

