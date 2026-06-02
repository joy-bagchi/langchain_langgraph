"""Composition root for layered platform services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_harness.agentic_os.context_service import DefaultContextService
from agentic_harness.agentic_os.dag_compiler import DefaultDagCompiler
from agentic_harness.agentic_os.dag_executor import DefaultDagExecutor
from agentic_harness.agentic_os.evaluation_service import BasicEvaluationService
from agentic_harness.agentic_os.guardrail_service import PassthroughGuardrailService
from agentic_harness.agentic_os.memory_service import (
    EphemeralMemoryService,
    FilesystemMemoryService,
    MemoryServiceSelection,
    SemanticMemoryService,
    StructuredMemoryService,
)
from agentic_harness.agentic_os.observability_service import (
    EventObservabilityService,
    resolve_langsmith_config,
)
from agentic_harness.agentic_os.security_service import PermissiveSecurityService
from agentic_harness.agentic_os.tool_service import RegisteredToolService
from agentic_harness.cognitive.service import DefaultCognitiveService
from agentic_harness.definitions.agent_service import YamlAgentDefinitionService
from agentic_harness.definitions.workflow_service import (
    MarkdownWorkflowDefinitionService,
    YamlDeclarativeWorkflowDefinitionService,
)
from agentic_harness.stores import WorkflowRunStore


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
    runtime_store: WorkflowRunStore
    guardrails: PassthroughGuardrailService
    observability: EventObservabilityService
    tools: RegisteredToolService
    evaluation: BasicEvaluationService
    security: PermissiveSecurityService


def build_platform_services(
    *,
    storage_root: str | Path | None = None,
    model_callable=None,
    memory_service_type: str = "filesystem",
    web_search_client=None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
    langsmith_client=None,
) -> PlatformServiceBundle:
    """Create the default in-process service bundle."""
    memory_selection = MemoryServiceSelection(
        service_type=memory_service_type,
        storage_root=storage_root,
        database_url=database_url,
    )
    if memory_selection.service_type == "ephemeral":
        memory_service = EphemeralMemoryService()
    elif memory_selection.service_type == "structured":
        memory_root = Path(storage_root or Path.cwd() / ".workflow_memory")
        memory_service = StructuredMemoryService(memory_root, database_url=database_url)
    elif memory_selection.service_type == "semantic":
        memory_root = Path(storage_root or Path.cwd() / ".workflow_memory")
        memory_service = SemanticMemoryService(memory_root, database_url=database_url)
    else:
        memory_root = Path(storage_root or Path.cwd() / ".workflow_memory")
        memory_service = FilesystemMemoryService(memory_root, database_url=database_url)
    runtime_store = WorkflowRunStore(
        Path(storage_root or Path.cwd() / ".workflow_memory"),
        database_url=database_url,
    )
    langsmith_config = resolve_langsmith_config(
        enabled=langsmith_tracing,
        api_key=langsmith_api_key,
        endpoint=langsmith_endpoint,
        project=langsmith_project,
        workspace_id=langsmith_workspace_id,
    )

    return PlatformServiceBundle(
        agent_definitions=YamlAgentDefinitionService(),
        workflow_definitions=MarkdownWorkflowDefinitionService(),
        declarative_workflow_definitions=YamlDeclarativeWorkflowDefinitionService(),
        dag_compiler=DefaultDagCompiler(),
        dag_executor=DefaultDagExecutor(),
        cognitive=DefaultCognitiveService(model_callable=model_callable),
        context=DefaultContextService(),
        memory=memory_service,
        runtime_store=runtime_store,
        guardrails=PassthroughGuardrailService(),
        observability=EventObservabilityService(
            langsmith_config=langsmith_config,
            langsmith_client=langsmith_client,
        ),
        tools=RegisteredToolService.with_defaults(web_search_client=web_search_client),
        evaluation=BasicEvaluationService(),
        security=PermissiveSecurityService(),
    )

