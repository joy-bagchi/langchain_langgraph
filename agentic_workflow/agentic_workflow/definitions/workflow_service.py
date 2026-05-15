"""Workflow definition service contract and markdown implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agentic_workflow.contracts import WorkflowDefinition
from agentic_workflow.markdown_workflow import load_workflow_definition
from agentic_workflow.shared.services import ServiceDescriptor


class WorkflowDefinitionService(Protocol):
    descriptor: ServiceDescriptor

    def load(self, path: str | Path) -> WorkflowDefinition:
        """Load a workflow definition."""


class MarkdownWorkflowDefinitionService:
    """Current workflow definition implementation backed by markdown DSL."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="workflow_definitions",
            implementation_id="markdown_workflow_definition_service",
            maturity="simple",
            capabilities=["markdown_dsl"],
        )

    def load(self, path: str | Path) -> WorkflowDefinition:
        return load_workflow_definition(path)
