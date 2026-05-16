"""Composition root for layered platform services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_harness.agentic_os.context_service import DefaultContextService
from agentic_harness.agentic_os.dag_compiler import DefaultDagCompiler
from agentic_harness.agentic_os.dag_executor import DefaultDagExecutor
from agentic_harness.agentic_os.evaluation_service import NullEvaluationService
from agentic_harness.agentic_os.guardrail_service import PassthroughGuardrailService
from agentic_harness.agentic_os.memory_service import (
    EphemeralMemoryService,
    FilesystemMemoryService,
    MemoryServiceSelection,
)
from agentic_harness.agentic_os.observability_service import EventObservabilityService
from agentic_harness.agentic_os.security_service import PermissiveSecurityService
from agentic_harness.agentic_os.tool_service import RegisteredToolService
from agentic_harness.cognitive.service import DefaultCognitiveService
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.definitions.workflow_service import (
    MarkdownWorkflowDefinitionService,
    YamlDeclarativeWorkflowDefinitionService,
)


@dataclass(slots=True)
class PlatformServiceBundle:
    """All runtime services required by the first platform slice."""

    agent_definitions: YamlAgentDefinitionService
    workflow_definitions: MarkdownWorkflowDefinitionService
    declarative_workflow_definitions: YamlDeclarativeWorkflowDefinitionService
    dag_compiler: DefaultDagCompiler
    dag_executor: DefaultDagExecutor
    cognitive: DefaultCognitiveService
    context: DefaultContextService
    memory: object
    guardrails: PassthroughGuardrailService
    observability: EventObservabilityService
    tools: RegisteredToolService
    evaluation: NullEvaluationService
    security: PermissiveSecurityService


def build_platform_services(
    *,
    storage_root: str | Path | None = None,
    model_callable=None,
    memory_service_type: str = "filesystem",
    web_search_client=None,
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
        agent_definitions=YamlAgentDefinitionService(),
        workflow_definitions=MarkdownWorkflowDefinitionService(),
        declarative_workflow_definitions=YamlDeclarativeWorkflowDefinitionService(),
        dag_compiler=DefaultDagCompiler(),
        dag_executor=DefaultDagExecutor(),
        cognitive=DefaultCognitiveService(model_callable=model_callable),
        context=DefaultContextService(),
        memory=memory_service,
        guardrails=PassthroughGuardrailService(),
        observability=EventObservabilityService(),
        tools=RegisteredToolService.with_defaults(web_search_client=web_search_client),
        evaluation=NullEvaluationService(),
        security=PermissiveSecurityService(),
    )

