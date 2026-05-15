"""LLM configuration and factory helpers for agentic_workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from agentic_workflow.contracts import WorkflowDefinition, WorkflowGraphState, WorkflowStep


Provider = Literal["none", "openai"]


@dataclass(slots=True)
class LLMConfig:
    """Configuration for prompt-step model execution."""

    provider: Provider = "none"
    model: str | None = None
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


def resolve_llm_config(
    *,
    workflow_definition: WorkflowDefinition | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LLMConfig:
    """Resolve LLM settings from explicit args, env vars, and workflow defaults."""
    resolved_provider = (
        provider
        or os.getenv("AGENTIC_WORKFLOW_LLM_PROVIDER")
        or ("openai" if model or os.getenv("AGENTIC_WORKFLOW_MODEL") else "none")
    ).strip().lower()
    if resolved_provider not in {"none", "openai"}:
        raise ValueError(
            f"Unsupported LLM provider '{resolved_provider}'. "
            "Supported providers: none, openai."
        )

    resolved_model = (
        model
        or os.getenv("AGENTIC_WORKFLOW_MODEL")
        or (workflow_definition.default_model if workflow_definition else None)
    )

    if temperature is None:
        raw_temperature = os.getenv("AGENTIC_WORKFLOW_TEMPERATURE")
        resolved_temperature = float(raw_temperature) if raw_temperature else 0.0
    else:
        resolved_temperature = temperature

    if resolved_provider == "none":
        return LLMConfig(provider="none", model=resolved_model, temperature=resolved_temperature)
    if not resolved_model:
        raise ValueError(
            "LLM provider is enabled but no model was configured. "
            "Pass --model, set AGENTIC_WORKFLOW_MODEL, or set default_model in the workflow."
        )
    return LLMConfig(
        provider="openai",
        model=resolved_model,
        temperature=resolved_temperature,
    )


def build_model_callable(config: LLMConfig):
    """Create a prompt executor callback from configuration."""
    if not config.enabled:
        return None

    if config.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenAI-backed prompt steps. "
                "Install it in the agentic_workflow environment."
            ) from exc

        model = ChatOpenAI(model=config.model, temperature=config.temperature)

        def invoke_model(
            prompt_text: str,
            step: WorkflowStep,
            state: WorkflowGraphState,
        ) -> Any:
            response = model.invoke(prompt_text)
            return getattr(response, "content", response)

        return invoke_model

    raise ValueError(f"Unsupported LLM provider '{config.provider}'.")
