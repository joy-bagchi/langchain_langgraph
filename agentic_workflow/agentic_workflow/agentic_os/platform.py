"""Composition root for layered platform services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_workflow.agentic_os.context_service import DefaultContextService
from agentic_workflow.agentic_os.evaluation_service import NullEvaluationService
from agentic_workflow.agentic_os.guardrail_service import PassthroughGuardrailService
from agentic_workflow.agentic_os.memory_service import (
    EphemeralMemoryService,
    FilesystemMemoryService,
    MemoryServiceSelection,
)
from agentic_workflow.agentic_os.observability_service import EventObservabilityService
from agentic_workflow.agentic_os.security_service import PermissiveSecurityService
from agentic_workflow.agentic_os.tool_service import NullToolService
from agentic_workflow.cognitive.service import DefaultCognitiveService
from agentic_workflow.definitions.workflow_service import MarkdownWorkflowDefinitionService


@dataclass(slots=True)
class PlatformServiceBundle:
    """All runtime services required by the first platform slice."""

    workflow_definitions: MarkdownWorkflowDefinitionService
    cognitive: DefaultCognitiveService
    context: DefaultContextService
    memory: object
    guardrails: PassthroughGuardrailService
    observability: EventObservabilityService
    tools: NullToolService
    evaluation: NullEvaluationService
    security: PermissiveSecurityService


def build_platform_services(
    *,
    storage_root: str | Path | None = None,
    model_callable=None,
    memory_service_type: str = "filesystem",
) -> PlatformServiceBundle:
    """Create the default in-process service bundle."""
    memory_selection = MemoryServiceSelection(
        service_type=memory_service_type,
        storage_root=storage_root,
    )
    if memory_selection.service_type == "ephemeral":
        memory_service = EphemeralMemoryService()
    else:
        memory_root = Path(storage_root or Path.cwd() / ".workflow_memory")
        memory_service = FilesystemMemoryService(memory_root)

    return PlatformServiceBundle(
        workflow_definitions=MarkdownWorkflowDefinitionService(),
        cognitive=DefaultCognitiveService(model_callable=model_callable),
        context=DefaultContextService(),
        memory=memory_service,
        guardrails=PassthroughGuardrailService(),
        observability=EventObservabilityService(),
        tools=NullToolService(),
        evaluation=NullEvaluationService(),
        security=PermissiveSecurityService(),
    )
