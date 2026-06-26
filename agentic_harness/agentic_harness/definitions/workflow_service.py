"""Workflow definition service contract and markdown implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agentic_harness.contracts import DeclarativeWorkflowDefinition, WorkflowDefinition
from agentic_harness.declarative_workflow import load_declarative_workflow_definition
from agentic_harness.markdown_workflow import load_workflow_definition
from agentic_harness.shared.services import ServiceDescriptor


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


class DeclarativeWorkflowDefinitionService(Protocol):
    descriptor: ServiceDescriptor

    def load(self, path: str | Path) -> DeclarativeWorkflowDefinition:
        """Load a declarative workflow definition."""


class YamlDeclarativeWorkflowDefinitionService:
    """Declarative workflow definition service backed by YAML."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="declarative_workflow_definitions",
            implementation_id="yaml_declarative_workflow_definition_service",
            maturity="simple",
            capabilities=["yaml_dag_workflow_dsl"],
        )

    def load(self, path: str | Path) -> DeclarativeWorkflowDefinition:
        return load_declarative_workflow_definition(path)

