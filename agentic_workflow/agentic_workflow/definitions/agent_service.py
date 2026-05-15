"""Agent definition service contract and yaml implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml

from agentic_workflow.contracts import AgentDefinition
from agentic_workflow.shared.services import ServiceDescriptor


class AgentDefinitionService(Protocol):
    descriptor: ServiceDescriptor

    def load(self, path: str | Path) -> AgentDefinition:
        """Load an agent definition."""


class YamlAgentDefinitionService:
    """Current agent definition implementation backed by YAML."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="agent_definitions",
            implementation_id="yaml_agent_definition_service",
            maturity="simple",
            capabilities=["yaml_agent_definition"],
        )

    def load(self, path: str | Path) -> AgentDefinition:
        agent_path = Path(path).resolve()
        payload: dict[str, Any] = yaml.safe_load(agent_path.read_text(encoding="utf-8")) or {}
        workflow_path = Path(payload["workflow_path"])
        if not workflow_path.is_absolute():
            workflow_path = (agent_path.parent / workflow_path).resolve()
        return AgentDefinition(
            agent_id=str(payload["agent_id"]),
            name=str(payload.get("name", payload["agent_id"])),
            role=str(payload["role"]),
            workflow_path=str(workflow_path),
            description=str(payload.get("description", "")),
            llm_provider=str(payload.get("llm_provider", "none")),
            model=payload.get("model"),
            temperature=float(payload.get("temperature", 0.0)),
            memory_service_type=str(payload.get("memory_service_type", "filesystem")),
            allowed_tools=list(payload.get("allowed_tools", [])),
            memory_namespace=payload.get("memory_namespace"),
            metadata={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "agent_id",
                    "name",
                    "role",
                    "workflow_path",
                    "description",
                    "llm_provider",
                    "model",
                    "temperature",
                    "memory_service_type",
                    "allowed_tools",
                    "memory_namespace",
                }
            },
        )
