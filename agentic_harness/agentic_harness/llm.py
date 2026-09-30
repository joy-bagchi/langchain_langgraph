"""LLM configuration and model execution for agentic_harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from jsonschema import Draft202012Validator

from agentic_harness.agentic_os.tool_service import ToolExecutionRequest, ToolService
from agentic_harness.contracts import WorkflowDefinition, WorkflowGraphState, WorkflowStep


Provider = Literal["none", "openai"]
OpenAIAPI = Literal["chat", "responses"]


@dataclass(slots=True)
class LLMConfig:
    """Configuration for prompt-step model execution."""

    provider: Provider = "none"
    model: str | None = None
    temperature: float = 0.0
    api: OpenAIAPI = "chat"
    reasoning_effort: str | None = None
    max_tool_rounds: int = 4
    max_tool_calls: int = 12
    request_timeout_seconds: float = 30.0
    max_request_retries: int = 1

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


@dataclass(slots=True)
class ModelExecutionResult:
    """Text for existing consumers plus typed Responses metadata for the ledger."""

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponsesExecutionError(RuntimeError):
    """A failed Responses turn with its completed, replayable transcript."""

    def __init__(self, message: str, transcript: list[dict[str, Any]],
                 metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metadata = {"responses": {"transcript": transcript, **(metadata or {})}}


def resolve_llm_config(
    *,
    workflow_definition: WorkflowDefinition | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api: str | None = None,
    reasoning_effort: str | None = None,
    max_tool_rounds: int | None = None,
    max_tool_calls: int | None = None,
    request_timeout_seconds: float | None = None,
    max_request_retries: int | None = None,
) -> LLMConfig:
    """Resolve LLM settings from explicit args, env vars, and workflow defaults."""
    resolved_provider = (
        provider
        or os.getenv("AGENTIC_HARNESS_LLM_PROVIDER")
        or ("openai" if model or os.getenv("AGENTIC_HARNESS_MODEL") else "none")
    ).strip().lower()
    if resolved_provider not in {"none", "openai"}:
        raise ValueError(
            f"Unsupported LLM provider '{resolved_provider}'. Supported providers: none, openai."
        )
    resolved_model = (
        model
        or os.getenv("AGENTIC_HARNESS_MODEL")
        or (workflow_definition.default_model if workflow_definition else None)
    )
    resolved_api = (api or os.getenv("AGENTIC_HARNESS_OPENAI_API") or "chat").strip().lower()
    if resolved_api not in {"chat", "responses"}:
        raise ValueError("OpenAI API mode must be 'chat' or 'responses'.")
    resolved_effort = reasoning_effort or os.getenv("AGENTIC_HARNESS_REASONING_EFFORT")
    rounds = max_tool_rounds if max_tool_rounds is not None else int(os.getenv("AGENTIC_HARNESS_MAX_TOOL_ROUNDS", "4"))
    calls = max_tool_calls if max_tool_calls is not None else int(os.getenv("AGENTIC_HARNESS_MAX_TOOL_CALLS", "12"))
    timeout = request_timeout_seconds if request_timeout_seconds is not None else float(os.getenv("AGENTIC_HARNESS_REQUEST_TIMEOUT_SECONDS", "30"))
    retries = max_request_retries if max_request_retries is not None else int(os.getenv("AGENTIC_HARNESS_MAX_REQUEST_RETRIES", "1"))
    if temperature is None:
        raw_temperature = os.getenv("AGENTIC_HARNESS_TEMPERATURE")
        resolved_temperature = float(raw_temperature) if raw_temperature else 0.0
    else:
        resolved_temperature = temperature
    if resolved_provider == "none":
        return LLMConfig(provider="none", model=resolved_model, temperature=resolved_temperature)
    if rounds < 0:
        raise ValueError("max_tool_rounds must be nonnegative.")
    if calls < 0 or timeout <= 0 or retries < 0:
        raise ValueError("Tool calls, request timeout, and request retries must have valid nonnegative bounds.")
    if resolved_effort and resolved_api != "responses":
        raise ValueError("reasoning_effort requires OpenAI Responses mode.")
    if resolved_effort and resolved_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"Unsupported reasoning_effort '{resolved_effort}'.")
    if resolved_effort in {"none", "minimal"} and str(resolved_model).startswith("gpt-6-astra"):
        raise ValueError("gpt-6-astra supports low, medium, high, xhigh, or max reasoning effort.")
    if not resolved_model:
        raise ValueError(
            "LLM provider is enabled but no model was configured. "
            "Pass --model, set AGENTIC_HARNESS_MODEL, or set default_model in the workflow."
        )
    return LLMConfig(
        provider="openai", model=resolved_model, temperature=resolved_temperature,
        api=resolved_api, reasoning_effort=resolved_effort, max_tool_rounds=rounds,
        max_tool_calls=calls, request_timeout_seconds=timeout, max_request_retries=retries,
    )


def _item_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Unsupported Responses item {type(item).__name__}.")


def _redact(value: Any) -> Any:
    """Remove credential-shaped fields before saving or replaying a transcript."""
    if isinstance(value, dict):
        cleaned = {
            key: "<redacted>" if any(word in key.lower() for word in ("api_key", "token", "password", "secret", "authorization"))
            else _redact(item)
            for key, item in value.items()
        }
        if value.get("type") == "function_call" and isinstance(value.get("arguments"), str):
            try:
                cleaned["arguments"] = json.dumps(_redact(json.loads(value["arguments"])))
            except ValueError:
                pass
        return cleaned
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _previous_transcript(state: WorkflowGraphState) -> list[dict[str, Any]]:
    for entry in reversed(state.get("step_history", [])):
        transcript = entry.get("metadata", {}).get("responses", {}).get("transcript")
        if transcript is not None:
            return list(transcript)
    return []


class ResponsesModel:
    """Stateless Responses adapter; the Harness ledger owns replay state."""

    def __init__(self, config: LLMConfig, client: Any) -> None:
        self.config = config
        self.client = client

    def __call__(
        self, prompt: str, step: WorkflowStep, state: WorkflowGraphState,
    ) -> str:
        """Keep the three-argument model callback usable for text-only callers."""
        return self.execute_prompt(prompt, step, state, None).output

    def execute_prompt(
        self, prompt: str, step: WorkflowStep, state: WorkflowGraphState,
        tool_service: ToolService | None, *, instructions: str | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> ModelExecutionResult:
        transcript = _previous_transcript(state)
        if runtime_context:
            transcript.append({"role": "developer", "content": "Run context: " + json.dumps(runtime_context, sort_keys=True)})
        transcript.append({"role": "user", "content": prompt})
        allowed = set(state.get("allowed_tools", []))
        registered = {tool.tool_id: tool for tool in tool_service.list_tools()} if tool_service else {}
        tools = [
            {"type": "function", "name": tool.tool_id, "description": tool.description,
             "parameters": tool.input_schema or {"type": "object", "properties": {}}}
            for tool_id, tool in registered.items() if tool_id in allowed
        ]
        rounds = 0
        call_count = 0
        response_ids: list[str] = []
        turns: list[dict[str, Any]] = []
        settings = {"model": self.config.model, "api": "responses",
                    "reasoning_effort": self.config.reasoning_effort,
                    "max_tool_rounds": self.config.max_tool_rounds,
                    "max_tool_calls": self.config.max_tool_calls,
                    "request_timeout_seconds": self.config.request_timeout_seconds,
                    "max_request_retries": self.config.max_request_retries,
                    "tools": _redact(tools), "store": False}

        def fail(message: str) -> ResponsesExecutionError:
            return ResponsesExecutionError(message, transcript, {
                **settings, "response_ids": response_ids, "turns": turns,
                "tool_rounds": rounds, "tool_calls": call_count,
            })

        while True:
            request: dict[str, Any] = {
                "model": self.config.model, "input": transcript, "store": False,
                "include": ["reasoning.encrypted_content"],
            }
            if instructions is not None:
                request["instructions"] = instructions
            if tools:
                request["tools"] = tools
            if self.config.reasoning_effort:
                request["reasoning"] = {"effort": self.config.reasoning_effort}
            if not (self.config.reasoning_effort or str(self.config.model).startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))):
                request["temperature"] = self.config.temperature
            requested_at = datetime.now(timezone.utc).isoformat()
            try:
                response = self.client.responses.create(**request)
            except Exception as exc:
                turns.append({"requested_at": requested_at, "status": "request_failed",
                              "error_type": type(exc).__name__})
                raise fail(f"Responses request failed: {type(exc).__name__}: {exc}") from exc
            response_id = getattr(response, "id", None)
            if response_id:
                response_ids.append(response_id)
            status = getattr(response, "status", None)
            turns.append({"requested_at": requested_at,
                          "received_at": datetime.now(timezone.utc).isoformat(),
                          "response_id": response_id, "status": status,
                          "created_at": getattr(response, "created_at", None),
                          "usage": _item_dict(response.usage) if getattr(response, "usage", None) is not None else None,
                          "input_item_count": len(transcript)})
            if status != "completed":
                raise fail(f"Responses returned status {status or 'missing'}.")
            try:
                items = [_item_dict(item) for item in response.output]
            except (AttributeError, TypeError, ValueError) as exc:
                raise fail("Responses returned malformed output items.") from exc
            transcript.extend(_redact(items))
            calls = [item for item in items if item.get("type") == "function_call"]
            unsupported = [item.get("type") for item in items if item.get("type") not in {"message", "reasoning", "function_call"}]
            if unsupported:
                raise fail(f"Unsupported Responses output item(s): {unsupported}.")
            refusals = [part for item in items if item.get("type") == "message"
                        for part in item.get("content", []) if part.get("type") == "refusal"]
            if refusals:
                raise fail("Responses refused the request.")
            if not calls:
                text_parts = [
                    part.get("text", "") for item in items if item.get("type") == "message"
                    for part in item.get("content", []) if part.get("type") == "output_text"
                ]
                if not text_parts:
                    raise fail("Responses returned no final text.")
                return ModelExecutionResult(
                    output="".join(text_parts),
                    metadata={"responses": {**settings, "response_ids": response_ids,
                        "turns": turns, "tool_rounds": rounds, "tool_calls": call_count,
                        "final_text": "".join(text_parts), "transcript": transcript}},
                )
            if rounds >= self.config.max_tool_rounds:
                raise fail(f"Responses exceeded {self.config.max_tool_rounds} tool rounds.")
            rounds += 1
            for call in calls:
                if call_count >= self.config.max_tool_calls:
                    raise fail(f"Responses exceeded {self.config.max_tool_calls} tool calls.")
                call_count += 1
                tool_id, call_id = call.get("name"), call.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise fail("Responses function call has no call_id.")
                if tool_id not in allowed:
                    raise fail(f"Agent is not allowed to use tool '{tool_id}'.")
                if tool_id not in registered:
                    raise fail(f"Tool '{tool_id}' is unavailable.")
                try:
                    arguments = json.loads(call.get("arguments", ""))
                except (TypeError, ValueError) as exc:
                    raise fail(f"Malformed arguments for tool '{tool_id}'.") from exc
                if not isinstance(arguments, dict):
                    raise fail(f"Malformed arguments for tool '{tool_id}': expected an object.")
                validation_errors = list(Draft202012Validator(
                    registered[tool_id].input_schema or {"type": "object"}
                ).iter_errors(arguments))
                if validation_errors:
                    raise fail(f"Malformed arguments for tool '{tool_id}': {validation_errors[0].message}")
                try:
                    result = tool_service.execute(ToolExecutionRequest(
                        tool_id=tool_id, arguments=arguments,
                        metadata={"run_id": state.get("run_id"), "workflow_id": state.get("workflow_id"),
                                  "step_id": step.step_id, "call_id": call_id},
                    ))
                except Exception as exc:
                    raise fail(f"Tool '{tool_id}' failed: {type(exc).__name__}: {exc}") from exc
                output = {"status": result.status, "output": result.output, "metadata": result.metadata}
                transcript.append({"type": "function_call_output", "call_id": call_id,
                                   "output": json.dumps(_redact(output), default=str)})
                if result.status != "succeeded":
                    reason = result.metadata.get("reason", result.status)
                    raise fail(f"Tool '{tool_id}' failed: {reason}")


def build_model_callable(config: LLMConfig, *, client: Any | None = None):
    """Create a prompt executor while preserving the legacy ChatOpenAI path."""
    if not config.enabled:
        return None
    if config.provider == "openai" and config.api == "responses":
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai is required for Responses mode.") from exc
            client = OpenAI(timeout=config.request_timeout_seconds,
                            max_retries=config.max_request_retries)
        return ResponsesModel(config, client)
    if config.provider == "openai" and config.api == "chat":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("langchain-openai is required for OpenAI-backed prompt steps.") from exc
        model = ChatOpenAI(model=config.model, temperature=config.temperature)

        def invoke_model(prompt_text: str, step: WorkflowStep, state: WorkflowGraphState) -> Any:
            response = model.invoke(prompt_text)
            return getattr(response, "content", response)

        return invoke_model
    raise ValueError(f"Unsupported LLM provider '{config.provider}' or API mode '{config.api}'.")
