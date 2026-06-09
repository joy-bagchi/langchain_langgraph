"""LangGraph-backed runtime for structured markdown workflows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agentic_harness.agentic_os.context_service import ContextServiceRequest
from agentic_harness.agentic_os.evaluation_service import EvaluationRequest
from agentic_harness.agentic_os.guardrail_service import GuardrailRequest
from agentic_harness.agentic_os.observability_service import ObservabilityRequest
from agentic_harness.agentic_os.platform import PlatformServiceBundle, build_platform_services
from agentic_harness.agentic_os.security_service import AuthorizationRequest
from agentic_harness.agentic_os.tool_service import ToolExecutionRequest
from agentic_harness.cognitive.service import PromptExecutionRequest
from agentic_harness.context import render_template, to_namespace
from agentic_harness.contracts import (
    AgentDefinition,
    AgentRuntimeProfile,
    ContextPolicy,
    MemoryLifecyclePolicy,
    MemoryQuery,
    MemoryRecord,
    StepExecutionResult,
    StepHistoryEntry,
    WorkflowDefinition,
    WorkflowGraphState,
    WorkflowStep,
    utc_now,
)
from agentic_harness.llm import build_model_callable, resolve_llm_config
from agentic_harness.stores import WorkflowRunStore
from agentic_harness.shared.services import ServiceEvent


Executor = Callable[[WorkflowStep, WorkflowGraphState, dict[str, Any]], StepExecutionResult]
ModelCallable = Callable[[str, WorkflowStep, WorkflowGraphState], Any]


def _deepcopy_dict(payload: dict[str, Any] | None) -> dict[str, Any]:
    return deepcopy(payload or {})


def _stringify_guardrail_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _trace_safe_payload(value: Any, *, max_string_chars: int = 1000, max_items: int = 10, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else f"{value[:max_string_chars]}...<truncated>"
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        result = {
            str(key): _trace_safe_payload(item, max_string_chars=max_string_chars, max_items=max_items, depth=depth + 1)
            for key, item in items
        }
        if len(value) > max_items:
            result["__truncated_keys__"] = len(value) - max_items
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _trace_safe_payload(item, max_string_chars=max_string_chars, max_items=max_items, depth=depth + 1)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            result.append(f"...<truncated {len(value) - max_items} items>")
        return result
    try:
        return _trace_safe_payload(json.loads(json.dumps(value, default=str)), max_string_chars=max_string_chars, max_items=max_items, depth=depth + 1)
    except Exception:
        rendered = str(value)
        return rendered if len(rendered) <= max_string_chars else f"{rendered[:max_string_chars]}...<truncated>"


def _trace_state_metadata(state: WorkflowGraphState) -> dict[str, Any]:
    return {
        "run_id": state.get("run_id"),
        "workflow_id": state.get("workflow_id"),
        "agent_id": state.get("agent_id"),
        "agent_role": state.get("agent_role"),
        "current_step": state.get("current_step"),
        "status": state.get("status"),
    }


def _build_guardrail_request_metadata(step: WorkflowStep, state: WorkflowGraphState) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if step.step_type == "tool":
        tool_id = str(step.metadata.get("tool_id", "")).strip()
        metadata["tool_id"] = tool_id
        argument_templates = step.metadata.get("arguments", {})
        if isinstance(argument_templates, dict):
            arguments = _render_tool_value(argument_templates, state)
            if "query_template" in step.metadata and "query" not in arguments:
                arguments["query"] = render_template(str(step.metadata["query_template"]), state)
            metadata["input_text"] = _stringify_guardrail_value(arguments)
    elif step.step_type == "collect":
        if "input_key" in step.metadata:
            metadata["input_text"] = _stringify_guardrail_value(
                state.get("input_payload", {}).get(step.metadata["input_key"])
            )
        else:
            metadata["input_text"] = _stringify_guardrail_value(state.get("input_payload", {}))
    else:
        metadata["input_text"] = render_template(step.prompt, state) if step.prompt else ""
    return metadata


def _build_guardrail_review_request(
    *,
    step: WorkflowStep,
    phase: str,
    reasons: list[str],
    candidate_output: Any,
) -> dict[str, Any]:
    phase_label = "pre" if phase == "pre_step" else "post"
    return {
        "step_id": step.step_id,
        "title": f"Guardrail Review: {step.title}",
        "review_type": f"guardrail_{phase_label}",
        "instructions": (
            f"Guardrail policy escalated this {phase.replace('_', '-')} review.\n\n"
            f"Reasons:\n- " + "\n- ".join(reasons)
        ),
        "candidate_output": candidate_output,
        "guardrail_reasons": list(reasons),
    }


def _build_evaluation_review_request(
    *,
    step: WorkflowStep,
    findings: list[str],
    candidate_output: Any,
) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": f"Evaluation Review: {step.title}",
        "review_type": "evaluation_post",
        "instructions": (
            "Evaluation policy escalated this output for review.\n\n"
            "Findings:\n- " + "\n- ".join(findings)
        ),
        "candidate_output": candidate_output,
        "evaluation_findings": list(findings),
    }


def _runtime_profile_defaults(profile_id: str | None) -> AgentRuntimeProfile:
    profile = (profile_id or "default").strip().lower()
    if profile == "durable_research":
        return AgentRuntimeProfile(
            profile_id=profile,
            context_policy=ContextPolicy(
                max_recent_history=5,
                max_memory_hits=5,
                max_working_notes_chars=1000,
                token_budget=1800,
            ),
            memory_policy=MemoryLifecyclePolicy(
                max_ephemeral_records=250,
                max_durable_records=20000,
            ),
            suspension_threshold_seconds=900,
        )
    if profile == "short_task":
        return AgentRuntimeProfile(
            profile_id=profile,
            context_policy=ContextPolicy(
                max_recent_history=2,
                max_memory_hits=2,
                max_working_notes_chars=300,
                token_budget=700,
            ),
            memory_policy=MemoryLifecyclePolicy(
                max_ephemeral_records=25,
                max_durable_records=500,
            ),
            suspension_threshold_seconds=120,
        )
    return AgentRuntimeProfile(profile_id="default")


def _evaluate_expression(expression: str, state: WorkflowGraphState) -> bool:
    """Evaluate a trusted workflow routing expression."""
    scope = {
        "input": to_namespace(dict(state.get("input_payload", {}))),
        "steps": to_namespace(
            {
                key: {"output": value}
                for key, value in state.get("step_outputs", {}).items()
            }
        ),
        "outputs": to_namespace(dict(state.get("named_outputs", {}))),
        "status": state.get("status"),
        "memory_hits": to_namespace(state.get("memory_hits", [])),
    }
    return bool(eval(expression, {"__builtins__": {}}, scope))


def _render_tool_value(value: Any, state: WorkflowGraphState) -> Any:
    if isinstance(value, str):
        return render_template(value, state)
    if isinstance(value, list):
        return [_render_tool_value(item, state) for item in value]
    if isinstance(value, dict):
        return {key: _render_tool_value(item, state) for key, item in value.items()}
    return value


def _default_executors(services: PlatformServiceBundle) -> dict[str, Executor]:
    def collect_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        if "input_key" in step.metadata:
            output = state.get("input_payload", {}).get(step.metadata["input_key"])
        else:
            output = render_template(step.prompt, state) or dict(state.get("input_payload", {}))
        return StepExecutionResult(output=output)

    def prompt_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        prompt_text = render_template(step.prompt, state)
        with services.observability.trace_span(
            f"prompt:{step.step_id}",
            run_type="llm",
            inputs={
                "prompt": _trace_safe_payload(prompt_text),
                "step_id": step.step_id,
                "step_type": step.step_type,
            },
            tags=["agentic_harness", "prompt", step.step_id],
            metadata=_trace_state_metadata(state),
        ) as prompt_span:
            response = services.cognitive.execute_prompt(
                PromptExecutionRequest(prompt=prompt_text, step=step, state=state)
            )
            if hasattr(prompt_span, "end"):
                prompt_span.end(
                    outputs={
                        "output": _trace_safe_payload(response.output),
                        "metadata": _trace_safe_payload(response.metadata),
                    }
                )
        return StepExecutionResult(output=response.output, metadata=response.metadata)

    def tool_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        tool_id = str(step.metadata.get("tool_id", "")).strip()
        if not tool_id:
            raise ValueError(f"Tool step '{step.step_id}' requires 'tool_id'.")

        allowed_tools = list(state.get("allowed_tools", []))
        if tool_id not in allowed_tools:
            raise PermissionError(
                f"Agent is not allowed to use tool '{tool_id}'."
            )

        argument_templates = step.metadata.get("arguments", {})
        if not isinstance(argument_templates, dict):
            raise ValueError(
                f"Tool step '{step.step_id}' expected 'arguments' to be a mapping."
            )
        arguments = _render_tool_value(argument_templates, state)
        if "query_template" in step.metadata and "query" not in arguments:
            arguments["query"] = render_template(str(step.metadata["query_template"]), state)

        with services.observability.trace_span(
            f"tool:{tool_id}",
            run_type="tool",
            inputs={
                "tool_id": tool_id,
                "arguments": _trace_safe_payload(arguments),
                "step_id": step.step_id,
            },
            tags=["agentic_harness", "tool", tool_id, step.step_id],
            metadata=_trace_state_metadata(state),
        ) as tool_span:
            response = services.tools.execute(
                ToolExecutionRequest(
                    tool_id=tool_id,
                    arguments=arguments,
                    metadata={
                        "step_id": step.step_id,
                        "workflow_id": state.get("workflow_id"),
                        "run_id": state.get("run_id"),
                    },
                )
            )
            if hasattr(tool_span, "end"):
                tool_span.end(
                    outputs={
                        "status": response.status,
                        "output": _trace_safe_payload(response.output),
                        "metadata": _trace_safe_payload(response.metadata),
                    }
                )
        if response.status != "succeeded":
            reason = response.metadata.get("reason", f"tool '{tool_id}' returned {response.status}")
            raise RuntimeError(str(reason))
        return StepExecutionResult(
            output=response.output,
            metadata={
                "tool_id": tool_id,
                "tool_status": response.status,
                **dict(response.metadata),
            },
        )

    def review_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        response = dict(state.get("review_responses", {})).get(step.step_id)
        review_target = state.get("step_history", [])[-1]["output"] if state.get("step_history") else None
        if not response:
            prompt_text = render_template(step.prompt, state)
            return StepExecutionResult(
                output=review_target,
                status="awaiting_review",
                awaiting_review=True,
                review_request={
                    "step_id": step.step_id,
                    "title": step.title,
                    "instructions": prompt_text,
                    "candidate_output": review_target,
                },
            )

        decision = str(response.get("decision", "")).strip().lower()
        if decision not in {"approved", "rejected"}:
            raise ValueError(
                f"Review step '{step.step_id}' expected decision to be 'approved' or 'rejected'."
            )
        next_step = step.approved_next if decision == "approved" else step.rejected_next
        return StepExecutionResult(
            output={"decision": decision, "notes": response.get("notes")},
            next_step=next_step,
        )

    def note_executor(
        step: WorkflowStep,
        state: WorkflowGraphState,
        _: dict[str, Any],
    ) -> StepExecutionResult:
        return StepExecutionResult(output=render_template(step.prompt, state))

    return {
        "collect": collect_executor,
        "prompt": prompt_executor,
        "tool": tool_executor,
        "human_review": review_executor,
        "note": note_executor,
    }


def compile_workflow(
    definition: WorkflowDefinition,
    *,
    services: PlatformServiceBundle,
    executors: dict[str, Executor] | None = None,
):
    """Compile a structured workflow into a LangGraph state machine."""
    executor_registry = _default_executors(services)
    executor_registry.update(executors or {})

    def load_run_context(state: WorkflowGraphState) -> WorkflowGraphState:
        events = list(state.get("events", []))
        if not events:
            events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
                event_type="run_started",
                payload={
                    "timestamp": utc_now(),
                    "run_id": state["run_id"],
                    "workflow_id": definition.workflow_id,
                },
            ))))
        return {"events": events}

    def retrieve_memory(state: WorkflowGraphState) -> WorkflowGraphState:
        if state.get("status") != "running" or not state.get("current_step"):
            return {}
        step = definition.steps[state["current_step"]]
        namespace = step.memory.namespace or definition.memory_namespace
        query_template = step.metadata.get("memory_query")
        query_text = render_template(query_template, state) if query_template else render_template(step.prompt, state)
        with services.observability.trace_span(
            f"memory_retrieval:{step.step_id}",
            run_type="retriever",
            inputs={
                "namespace": namespace,
                "query_text": _trace_safe_payload(query_text),
            },
            tags=["agentic_harness", "memory", "retrieve", step.step_id],
            metadata=_trace_state_metadata(state),
        ) as memory_span:
            results = services.memory.recall(
                MemoryQuery(namespace=namespace, text=query_text, max_results=5)
            )
            if hasattr(memory_span, "end"):
                memory_span.end(
                    outputs={
                        "match_count": len(results),
                        "matches": _trace_safe_payload(
                            [
                                {
                                    "memory_type": item.record.memory_type,
                                    "content": item.record.content,
                                    "score": item.score,
                                }
                                for item in results
                            ]
                        ),
                    }
                )
        events = list(state.get("events", []))
        events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
            event_type="memory_retrieved",
            payload={
                "timestamp": utc_now(),
                "step_id": step.step_id,
                "match_count": len(results),
            },
        ))))
        return {
            "memory_hits": [item.to_dict() for item in results],
            "events": events,
        }

    def prepare_context(state: WorkflowGraphState) -> WorkflowGraphState:
        if state.get("status") != "running" or not state.get("current_step"):
            return {}
        step = definition.steps[state["current_step"]]
        with services.observability.trace_span(
            f"context_preparation:{step.step_id}",
            run_type="chain",
            inputs={
                "memory_hit_count": len(state.get("memory_hits", [])),
                "history_count": len(state.get("step_history", [])),
            },
            tags=["agentic_harness", "context", step.step_id],
            metadata=_trace_state_metadata(state),
        ) as context_span:
            snapshot = services.context.assemble_context(
                ContextServiceRequest(
                    workflow_definition=definition,
                    step=step,
                    state=state,
                )
            )
            if hasattr(context_span, "end"):
                context_span.end(
                    outputs={
                        "context_brief": _trace_safe_payload(snapshot.context_brief),
                        "compaction_decision": _trace_safe_payload(snapshot.compaction_decision),
                        "recent_history_count": len(snapshot.recent_history),
                        "memory_hit_count": len(snapshot.raw_memory_hits),
                    }
                )
        events = list(state.get("events", []))
        events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
            event_type="context_prepared",
            payload={
                "timestamp": utc_now(),
                "step_id": step.step_id,
                "recent_history_count": len(snapshot.recent_history),
                "compacted_history_present": bool(snapshot.compacted_history),
                "memory_hit_count": len(snapshot.raw_memory_hits),
            },
        ))))
        return {
            "active_context": snapshot.to_dict(),
            "compaction_decision": dict(snapshot.compaction_decision),
            "events": events,
        }

    def execute_step(state: WorkflowGraphState) -> WorkflowGraphState:
        step_id = state.get("current_step")
        if not step_id:
            return {}
        step = definition.steps[step_id]
        retry_counts = dict(state.get("retry_counts", {}))
        events = list(state.get("events", []))
        review_responses = dict(state.get("review_responses", {}))
        attempt = retry_counts.get(step_id, 0) + 1

        def _failure_state(exc: Exception, *, retryable: bool) -> WorkflowGraphState:
            should_retry = retryable and attempt <= step.max_retries
            retry_counts[step_id] = attempt
            history = list(state.get("step_history", []))
            history.append(
                StepHistoryEntry(
                    step_id=step_id,
                    status="failed" if not should_retry else "retrying",
                    output=None,
                    next_step=step_id if should_retry else None,
                    attempt=attempt,
                    metadata={"error": str(exc)},
                ).to_dict()
            )
            events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
                event_type="step_error",
                payload={
                    "timestamp": utc_now(),
                    "step_id": step_id,
                    "attempt": attempt,
                    "error": str(exc),
                },
            ))))
            return {
                "retry_counts": retry_counts,
                "step_history": history,
                "events": events,
                "execution_outcome": {
                    "route": "checkpoint",
                    "completed_step_id": step_id,
                },
                "status": "running" if should_retry else "failed",
                "current_step": step_id if should_retry else None,
                "last_error": str(exc),
                "review_responses": review_responses,
            }

        guardrail_response = review_responses.get(step_id)
        skip_post_guardrail = False
        skip_evaluation = False
        if isinstance(guardrail_response, dict) and guardrail_response.get("review_type") in {"guardrail_post", "evaluation_post"}:
            review_payload = dict(guardrail_response)
            review_responses.pop(step_id, None)
            decision = str(review_payload.get("decision", "")).strip().lower()
            review_type = str(review_payload.get("review_type", "")).strip().lower()
            if decision == "rejected":
                return _failure_state(
                    PermissionError(f"{review_type} review rejected output"),
                    retryable=False,
                )
            if decision == "approved":
                result = StepExecutionResult(
                    output=review_payload.get("candidate_output"),
                    metadata={
                        **({ "guardrail_review": "approved" } if review_type == "guardrail_post" else {}),
                        f"{review_type}_review": "approved",
                        "guardrail_reasons": list(review_payload.get("guardrail_reasons", [])),
                        "evaluation_findings": list(review_payload.get("evaluation_findings", [])),
                    },
                )
                skip_post_guardrail = True
                skip_evaluation = True
            else:
                return _failure_state(
                    PermissionError(f"{review_type} review expected approved or rejected"),
                    retryable=False,
                )
        else:
            result = None

        authz = services.security.authorize(
            AuthorizationRequest(
                capability=f"step:{step.step_type}",
                metadata={"step_id": step.step_id, "workflow_id": definition.workflow_id},
            )
        )
        if not authz.allowed:
            return _failure_state(PermissionError(authz.reason), retryable=False)

        if result is None:
            pre_guardrail_metadata = _build_guardrail_request_metadata(step, state)
            with services.observability.trace_span(
                f"guardrail_pre:{step.step_id}",
                run_type="chain",
                inputs=_trace_safe_payload(pre_guardrail_metadata),
                tags=["agentic_harness", "guardrail", "pre", step.step_id],
                metadata=_trace_state_metadata(state),
            ) as pre_guardrail_span:
                pre_guardrail = services.guardrails.evaluate(
                    GuardrailRequest(
                        phase="pre_step",
                        step=step,
                        state=state,
                        metadata=pre_guardrail_metadata,
                    )
                )
                if hasattr(pre_guardrail_span, "end"):
                    pre_guardrail_span.end(
                        outputs={
                            "allowed": pre_guardrail.allowed,
                            "action": pre_guardrail.action,
                            "reasons": _trace_safe_payload(pre_guardrail.reasons),
                        }
                    )
            if not pre_guardrail.allowed:
                return _failure_state(
                    PermissionError("; ".join(pre_guardrail.reasons) or "pre-step guardrail blocked execution"),
                    retryable=False,
                )

            try:
                executor = executor_registry.get(step.step_type)
                if executor is None:
                    raise ValueError(
                        f"No executor registered for step type '{step.step_type}'."
                    )
                with services.observability.trace_span(
                    f"step_executor:{step.step_id}",
                    run_type="chain",
                    inputs={
                        "step_type": step.step_type,
                        "step_id": step.step_id,
                        "active_context": _trace_safe_payload(state.get("active_context", {})),
                    },
                    tags=["agentic_harness", "step", step.step_type, step.step_id],
                    metadata=_trace_state_metadata(state),
                ) as executor_span:
                    result = executor(step, state, {})
                    if hasattr(executor_span, "end"):
                        executor_span.end(
                            outputs={
                                "status": result.status,
                                "output": _trace_safe_payload(result.output),
                                "awaiting_review": result.awaiting_review,
                                "next_step": result.next_step,
                                "metadata": _trace_safe_payload(result.metadata),
                            }
                        )
            except Exception as exc:
                return _failure_state(exc, retryable=True)

        step_outputs = dict(state.get("step_outputs", {}))
        named_outputs = dict(state.get("named_outputs", {}))
        if result.output is not None:
            step_outputs[step_id] = result.output
            if step.output_key:
                named_outputs[step.output_key] = result.output
        post_guardrail = None
        if not skip_post_guardrail:
            post_guardrail_metadata = {"output_text": _stringify_guardrail_value(result.output)}
            with services.observability.trace_span(
                f"guardrail_post:{step.step_id}",
                run_type="chain",
                inputs=_trace_safe_payload(post_guardrail_metadata),
                tags=["agentic_harness", "guardrail", "post", step.step_id],
                metadata=_trace_state_metadata(state),
            ) as post_guardrail_span:
                post_guardrail = services.guardrails.evaluate(
                    GuardrailRequest(
                        phase="post_step",
                        step=step,
                        state={**state, "step_outputs": step_outputs, "named_outputs": named_outputs},
                        candidate_output=result.output,
                        metadata=post_guardrail_metadata,
                    )
                )
                if hasattr(post_guardrail_span, "end"):
                    post_guardrail_span.end(
                        outputs={
                            "allowed": post_guardrail.allowed,
                            "action": post_guardrail.action,
                            "reasons": _trace_safe_payload(post_guardrail.reasons),
                        }
                    )
        if post_guardrail is not None and not post_guardrail.allowed:
            if post_guardrail.action == "escalate":
                history = list(state.get("step_history", []))
                history.append(
                    StepHistoryEntry(
                        step_id=step_id,
                        status="awaiting_review",
                        output=result.output,
                        next_step=step_id,
                        attempt=attempt,
                        metadata={
                            "guardrail_action": "escalate",
                            "guardrail_reasons": list(post_guardrail.reasons),
                        },
                    ).to_dict()
                )
                events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
                    event_type="guardrail_review_requested",
                    payload={
                        "timestamp": utc_now(),
                        "step_id": step_id,
                        "phase": "post_step",
                        "reasons": list(post_guardrail.reasons),
                    },
                ))))
                return {
                    "step_outputs": step_outputs,
                    "named_outputs": named_outputs,
                    "step_history": history,
                    "events": events,
                    "pending_review": _build_guardrail_review_request(
                        step=step,
                        phase="post_step",
                        reasons=post_guardrail.reasons,
                        candidate_output=result.output,
                    ),
                    "status": "awaiting_review",
                    "execution_outcome": {
                        "route": "checkpoint",
                        "completed_step_id": step_id,
                    },
                    "last_error": None,
                    "review_responses": review_responses,
                }
            return _failure_state(
                PermissionError("; ".join(post_guardrail.reasons) or "post-step guardrail blocked output"),
                retryable=False,
            )
        evaluation = None
        if not skip_evaluation:
            evaluation_payload = {
                "step_id": step_id,
                "status": result.status,
                "output": result.output,
                "output_text": _stringify_guardrail_value(result.output),
                "attempt": attempt,
                "max_retries": step.max_retries,
                "workflow_metadata": dict(definition.metadata),
                "step_metadata": dict(step.metadata),
                "agent_metadata": dict(state.get("agent_metadata", {})),
                "workflow_id": definition.workflow_id,
                "step_history": list(state.get("step_history", [])),
                "named_outputs": dict(named_outputs),
                "active_context": dict(state.get("active_context", {})),
            }
            with services.observability.trace_span(
                f"evaluation:{step.step_id}",
                run_type="chain",
                inputs=_trace_safe_payload(evaluation_payload),
                tags=["agentic_harness", "evaluation", step.step_id],
                metadata=_trace_state_metadata(state),
            ) as evaluation_span:
                evaluation = services.evaluation.evaluate(
                    EvaluationRequest(
                        phase="step",
                        payload=evaluation_payload,
                    )
                )
                if hasattr(evaluation_span, "end"):
                    evaluation_span.end(
                        outputs={
                            "status": evaluation.status,
                            "action": evaluation.action,
                            "findings": _trace_safe_payload(evaluation.findings),
                            "score": evaluation.score,
                        }
                    )
        if evaluation is not None and evaluation.action == "retry":
            reason = "; ".join(evaluation.findings) or "evaluation requested retry"
            return _failure_state(PermissionError(reason), retryable=True)
        if evaluation is not None and evaluation.action == "escalate":
            history = list(state.get("step_history", []))
            history.append(
                StepHistoryEntry(
                    step_id=step_id,
                    status="awaiting_review",
                    output=result.output,
                    next_step=step_id,
                    attempt=attempt,
                    metadata={
                        "evaluation_action": "escalate",
                        "evaluation_findings": list(evaluation.findings),
                    },
                ).to_dict()
            )
            events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
                event_type="evaluation_review_requested",
                payload={
                    "timestamp": utc_now(),
                    "step_id": step_id,
                    "findings": list(evaluation.findings),
                },
            ))))
            return {
                "step_outputs": step_outputs,
                "named_outputs": named_outputs,
                "step_history": history,
                "events": events,
                "pending_review": _build_evaluation_review_request(
                    step=step,
                    findings=evaluation.findings,
                    candidate_output=result.output,
                ),
                "status": "awaiting_review",
                "execution_outcome": {
                    "route": "checkpoint",
                    "completed_step_id": step_id,
                },
                "last_error": None,
                "review_responses": review_responses,
            }
        if evaluation is not None and evaluation.action == "fail":
            reason = "; ".join(evaluation.findings) or "evaluation failed output"
            return _failure_state(PermissionError(reason), retryable=False)

        if result.awaiting_review:
            history = list(state.get("step_history", []))
            history.append(
                StepHistoryEntry(
                    step_id=step_id,
                    status="awaiting_review",
                    output=result.output,
                    next_step=step_id,
                    attempt=attempt,
                ).to_dict()
            )
            events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
                event_type="review_requested",
                payload={
                    "timestamp": utc_now(),
                    "step_id": step_id,
                },
            ))))
            return {
                "step_outputs": step_outputs,
                "named_outputs": named_outputs,
                "step_history": history,
                "events": events,
                "pending_review": result.review_request,
                "status": "awaiting_review",
                "execution_outcome": {
                    "route": "checkpoint",
                    "completed_step_id": step_id,
                },
                "last_error": None,
                "review_responses": review_responses,
            }

        next_step = result.next_step
        if next_step is None:
            for branch in step.branches:
                if _evaluate_expression(branch.when, {**state, "step_outputs": step_outputs, "named_outputs": named_outputs}):
                    next_step = branch.next_step
                    break
        if next_step is None:
            next_step = step.next_step

        history = list(state.get("step_history", []))
        history.append(
            StepHistoryEntry(
                step_id=step_id,
                status=result.status,
                output=result.output,
                next_step=next_step,
                attempt=attempt,
                metadata={
                    **dict(result.metadata),
                    "evaluation": {
                        "status": evaluation.status if evaluation is not None else "approved_review",
                        "action": evaluation.action if evaluation is not None else "allow",
                        "findings": list(evaluation.findings) if evaluation is not None else [],
                        "score": evaluation.score if evaluation is not None else 1.0,
                    },
                },
            ).to_dict()
        )
        events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
            event_type="step_executed",
            payload={
                "timestamp": utc_now(),
                "step_id": step_id,
                "attempt": attempt,
                "next_step": next_step,
                "cognitive_service": services.cognitive.descriptor.implementation_id,
                "memory_service": services.memory.descriptor.implementation_id,
            },
        ))))

        status = "running" if next_step else "completed"
        route = "write_memory" if step.memory.enabled and result.output is not None else "checkpoint"
        return {
            "retry_counts": retry_counts,
            "step_outputs": step_outputs,
            "named_outputs": named_outputs,
            "step_history": history,
            "events": events,
            "pending_review": None,
            "status": status,
            "current_step": next_step,
            "execution_outcome": {
                "route": route,
                "completed_step_id": step_id,
                "next_step": next_step,
                "output": result.output,
                "memory_content": result.memory_content,
            },
            "last_error": None,
            "review_responses": {key: value for key, value in review_responses.items() if key != step_id},
        }

    def write_memory(state: WorkflowGraphState) -> WorkflowGraphState:
        outcome = dict(state.get("execution_outcome", {}))
        step_id = outcome.get("completed_step_id")
        if not step_id:
            return {}
        step = definition.steps[step_id]
        if not step.memory.enabled:
            return {}

        state_for_render: WorkflowGraphState = {**state, "current_step": step_id}
        content = outcome.get("memory_content")
        if content is None:
            content = render_template(
                step.memory.template,
                state_for_render,
                extra={"step_output": outcome.get("output")},
            )
        if not content:
            content = json.dumps(outcome.get("output"), sort_keys=True)

        with services.observability.trace_span(
            f"memory_write:{step_id}",
            run_type="tool",
            inputs={
                "namespace": step.memory.namespace or definition.memory_namespace,
                "memory_type": step.memory.memory_type,
                "content": _trace_safe_payload(str(content)),
            },
            tags=["agentic_harness", "memory", "write", step_id],
            metadata=_trace_state_metadata(state),
        ) as memory_write_span:
            record = services.memory.remember(
                MemoryRecord.create(
                    namespace=step.memory.namespace or definition.memory_namespace,
                    memory_type=step.memory.memory_type,
                    content=str(content),
                    source_run_id=state["run_id"],
                    source_step_id=step_id,
                    ttl_days=step.memory.ttl_days,
                    metadata=step.memory.metadata,
                )
            )
            if hasattr(memory_write_span, "end"):
                memory_write_span.end(
                    outputs={
                        "record_id": record.record_id,
                        "namespace": record.namespace,
                        "memory_type": record.memory_type,
                    }
                )
        history = list(state.get("step_history", []))
        if history:
            history[-1]["memory_record_ids"] = list(history[-1].get("memory_record_ids", []))
            history[-1]["memory_record_ids"].append(record.record_id)
        events = list(state.get("events", []))
        events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
            event_type="memory_written",
            payload={
                "timestamp": utc_now(),
                "step_id": step_id,
                "record_id": record.record_id,
            },
        ))))
        outcome["memory_record_id"] = record.record_id
        outcome["route"] = "checkpoint"
        return {
            "step_history": history,
            "events": events,
            "execution_outcome": outcome,
        }

    def checkpoint_run(state: WorkflowGraphState) -> WorkflowGraphState:
        next_index = int(state.get("checkpoint_index", 0)) + 1
        with services.observability.trace_span(
            f"checkpoint:{next_index}",
            run_type="chain",
            inputs={
                "checkpoint_index": next_index,
                "status": state.get("status"),
                "current_step": state.get("current_step"),
            },
            tags=["agentic_harness", "checkpoint"],
            metadata=_trace_state_metadata(state),
        ) as checkpoint_span:
            if hasattr(checkpoint_span, "end"):
                checkpoint_span.end(
                    outputs={
                        "checkpoint_index": next_index,
                        "status": state.get("status"),
                        "current_step": state.get("current_step"),
                    }
                )
        events = list(state.get("events", []))
        events.append(services.observability.record(ObservabilityRequest(event=ServiceEvent(
            event_type="checkpoint",
            payload={
                "timestamp": utc_now(),
                "checkpoint_index": next_index,
                "status": state.get("status"),
                "current_step": state.get("current_step"),
            },
        ))))
        return {"checkpoint_index": next_index, "events": events}

    def route_after_execute(state: WorkflowGraphState) -> str:
        if state.get("execution_outcome", {}).get("route") == "write_memory":
            return "write_memory"
        return "checkpoint"

    def route_after_checkpoint(state: WorkflowGraphState) -> str:
        if state.get("status") == "running" and state.get("current_step"):
            return "continue"
        return "end"

    builder = StateGraph(WorkflowGraphState)
    builder.add_node("load_run_context", load_run_context)
    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("execute_step", execute_step)
    builder.add_node("write_memory", write_memory)
    builder.add_node("checkpoint_run", checkpoint_run)

    builder.add_edge(START, "load_run_context")
    builder.add_edge("load_run_context", "retrieve_memory")
    builder.add_edge("retrieve_memory", "prepare_context")
    builder.add_edge("prepare_context", "execute_step")
    builder.add_conditional_edges(
        "execute_step",
        route_after_execute,
        {
            "write_memory": "write_memory",
            "checkpoint": "checkpoint_run",
        },
    )
    builder.add_edge("write_memory", "checkpoint_run")
    builder.add_conditional_edges(
        "checkpoint_run",
        route_after_checkpoint,
        {
            "continue": "retrieve_memory",
            "end": END,
        },
    )
    return builder.compile()


class WorkflowRunner:
    """High-level runner for structured markdown workflow definitions."""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        storage_root: str | Path | None = None,
        executors: dict[str, Executor] | None = None,
        model_callable: ModelCallable | None = None,
        services: PlatformServiceBundle | None = None,
        memory_service_type: str = "filesystem",
    ) -> None:
        self.definition = definition
        self.storage_root = Path(storage_root or Path.cwd() / ".workflow_memory")
        self.services = services or build_platform_services(
            storage_root=self.storage_root,
            model_callable=model_callable,
            memory_service_type=memory_service_type,
        )
        self.run_store = self.services.runtime_store if hasattr(self.services, "runtime_store") else WorkflowRunStore(self.storage_root)
        self.graph = compile_workflow(
            definition,
            services=self.services,
            executors=executors,
        )

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "metadata": {
                "workflow_id": self.definition.workflow_id,
                "workflow_title": self.definition.title,
                "run_id": run_id,
                "memory_namespace": self.definition.memory_namespace,
                "services": {
                    "cognitive": self.services.cognitive.descriptor.implementation_id,
                    "context": self.services.context.descriptor.implementation_id,
                    "memory": self.services.memory.descriptor.implementation_id,
                },
            },
            "tags": ["agentic_harness", self.definition.workflow_id],
        }

    def _persist(self, state: dict[str, Any]) -> None:
        self.run_store.save_state(state)
        self.run_store.save_manifest(
            state["run_id"],
            {
                "workflow_id": self.definition.workflow_id,
                "workflow_path": self.definition.workflow_path,
                "storage_root": str(self.storage_root),
            },
        )

    def _bootstrap_state(
        self,
        *,
        input_payload: dict[str, Any],
        run_id: str | None,
        review_responses: dict[str, dict[str, Any]] | None = None,
        initial_state_overrides: dict[str, Any] | None = None,
    ) -> WorkflowGraphState:
        resolved_run_id = run_id or str(uuid4())
        state = WorkflowGraphState(
            run_id=resolved_run_id,
            agent_id=None,
            agent_name=None,
            agent_role=None,
            agent_metadata={},
            invocation_id=resolved_run_id,
            runtime_profile="default",
            allowed_tools=[],
            workflow_id=self.definition.workflow_id,
            workflow_title=self.definition.title,
            workflow_path=self.definition.workflow_path or "",
            status="running",
            current_step=self.definition.entry_step,
            input_payload=_deepcopy_dict(input_payload),
            named_outputs={},
            step_outputs={},
            working_notes=[],
            memory_hits=[],
            step_history=[],
            pending_review=None,
            review_responses=_deepcopy_dict(review_responses),
            retry_counts={},
            events=[],
            execution_outcome={},
            active_context={},
            context_policy=ContextPolicy().to_dict(),
            memory_policy=MemoryLifecyclePolicy().to_dict(),
            compaction_decision=None,
            checkpoint_index=0,
            last_error=None,
        )
        if initial_state_overrides:
            state.update(initial_state_overrides)
        return state

    def start(
        self,
        input_payload: dict[str, Any],
        *,
        run_id: str | None = None,
        review_responses: dict[str, dict[str, Any]] | None = None,
        initial_state_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._bootstrap_state(
            input_payload=input_payload,
            run_id=run_id,
            review_responses=review_responses,
            initial_state_overrides=initial_state_overrides,
        )
        latest_state = dict(state)
        with self.services.observability.trace_context(
            project_name=None,
            tags=["agentic_harness", "workflow", self.definition.workflow_id],
            metadata={
                "run_id": latest_state["run_id"],
                "workflow_id": self.definition.workflow_id,
                "agent_id": latest_state.get("agent_id"),
                "agent_role": latest_state.get("agent_role"),
            },
        ):
            with self.services.observability.trace_span(
                f"workflow_run:{self.definition.workflow_id}",
                run_type="chain",
                inputs={
                    "input_payload": _trace_safe_payload(input_payload),
                    "workflow_id": self.definition.workflow_id,
                    "workflow_title": self.definition.title,
                },
                tags=["agentic_harness", "workflow", self.definition.workflow_id],
                metadata={
                    "run_id": latest_state["run_id"],
                    "workflow_id": self.definition.workflow_id,
                    "agent_id": latest_state.get("agent_id"),
                    "agent_role": latest_state.get("agent_role"),
                },
            ) as workflow_span:
                for latest_state in self.graph.stream(
                    state,
                    config=self._config(latest_state["run_id"]),
                    stream_mode="values",
                ):
                    self._persist(latest_state)
                self._persist(latest_state)
                if hasattr(workflow_span, "end"):
                    workflow_span.end(
                        outputs={
                            "status": latest_state.get("status"),
                            "current_step": latest_state.get("current_step"),
                            "pending_review": _trace_safe_payload(latest_state.get("pending_review")),
                            "last_error": latest_state.get("last_error"),
                            "named_outputs": _trace_safe_payload(latest_state.get("named_outputs", {})),
                        }
                    )
        self.services.observability.flush()
        return latest_state

    def resume(
        self,
        run_id: str,
        *,
        decision: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        state = self.run_store.load_state(run_id)
        if decision:
            pending_review = state.get("pending_review") or {}
            step_id = pending_review.get("step_id") or state.get("current_step")
            if step_id:
                review_responses = dict(state.get("review_responses", {}))
                review_responses[step_id] = {
                    "decision": decision,
                    "notes": notes,
                    "review_type": pending_review.get("review_type"),
                    "candidate_output": pending_review.get("candidate_output"),
                    "guardrail_reasons": list(pending_review.get("guardrail_reasons", [])),
                }
                state["review_responses"] = review_responses
            state["status"] = "running"
            state["pending_review"] = None

        latest_state = dict(state)
        with self.services.observability.trace_context(
            project_name=None,
            tags=["agentic_harness", "workflow_resume", self.definition.workflow_id],
            metadata={
                "run_id": run_id,
                "workflow_id": self.definition.workflow_id,
                "decision": decision,
            },
        ):
            with self.services.observability.trace_span(
                f"workflow_resume:{self.definition.workflow_id}",
                run_type="chain",
                inputs={
                    "decision": decision,
                    "notes": _trace_safe_payload(notes),
                    "workflow_id": self.definition.workflow_id,
                },
                tags=["agentic_harness", "workflow_resume", self.definition.workflow_id],
                metadata={
                    "run_id": run_id,
                    "workflow_id": self.definition.workflow_id,
                },
            ) as workflow_resume_span:
                for latest_state in self.graph.stream(
                    state,
                    config=self._config(run_id),
                    stream_mode="values",
                ):
                    self._persist(latest_state)
                self._persist(latest_state)
                if hasattr(workflow_resume_span, "end"):
                    workflow_resume_span.end(
                        outputs={
                            "status": latest_state.get("status"),
                            "current_step": latest_state.get("current_step"),
                            "pending_review": _trace_safe_payload(latest_state.get("pending_review")),
                            "last_error": latest_state.get("last_error"),
                            "named_outputs": _trace_safe_payload(latest_state.get("named_outputs", {})),
                        }
                    )
        self.services.observability.flush()
        return latest_state


def start_workflow(
    path: str | Path,
    input_payload: dict[str, Any],
    *,
    run_id: str | None = None,
    storage_root: str | Path | None = None,
    executors: dict[str, Executor] | None = None,
    model_callable: ModelCallable | None = None,
    services: PlatformServiceBundle | None = None,
    memory_service_type: str = "filesystem",
    initial_state_overrides: dict[str, Any] | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Load a workflow from disk and execute it."""
    service_bundle = services or build_platform_services(
        storage_root=storage_root,
        model_callable=model_callable,
        memory_service_type=memory_service_type,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    definition = service_bundle.workflow_definitions.load(path)
    runner = WorkflowRunner(
        definition,
        storage_root=storage_root,
        executors=executors,
        model_callable=model_callable,
        services=service_bundle,
        memory_service_type=memory_service_type,
    )
    return runner.start(
        input_payload,
        run_id=run_id,
        initial_state_overrides=initial_state_overrides,
    )


def resume_workflow(
    run_id: str,
    *,
    storage_root: str | Path | None = None,
    decision: str | None = None,
    notes: str | None = None,
    executors: dict[str, Executor] | None = None,
    model_callable: ModelCallable | None = None,
    services: PlatformServiceBundle | None = None,
    memory_service_type: str = "filesystem",
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Resume a persisted workflow run."""
    storage_path = Path(storage_root or Path.cwd() / ".workflow_memory")
    run_store = WorkflowRunStore(storage_path)
    state = run_store.load_state(run_id)
    service_bundle = services or build_platform_services(
        storage_root=storage_path,
        model_callable=model_callable,
        memory_service_type=memory_service_type,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    definition = service_bundle.workflow_definitions.load(state["workflow_path"])
    runner = WorkflowRunner(
        definition,
        storage_root=storage_path,
        executors=executors,
        model_callable=model_callable,
        services=service_bundle,
        memory_service_type=memory_service_type,
    )
    return runner.resume(run_id, decision=decision, notes=notes)


def inspect_run(
    run_id: str,
    *,
    storage_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the persisted state for a workflow run."""
    storage_path = Path(storage_root or Path.cwd() / ".workflow_memory")
    return WorkflowRunStore(storage_path).inspect(run_id)


def run_agent_workflow(
    agent_path: str | Path,
    input_payload: dict[str, Any],
    *,
    run_id: str | None = None,
    storage_root: str | Path | None = None,
    services: PlatformServiceBundle | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Load an agent definition, resolve its workflow, and execute it."""
    bootstrap_services = services or build_platform_services(
        storage_root=storage_root,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    agent_definition = bootstrap_services.agent_definitions.load(agent_path)
    runtime_profile = _runtime_profile_defaults(agent_definition.runtime_profile)
    if services is None:
        llm_config = resolve_llm_config(
            provider=agent_definition.llm_provider,
            model=agent_definition.model,
            temperature=agent_definition.temperature,
        )
        bound_services = build_platform_services(
            storage_root=storage_root,
            model_callable=build_model_callable(llm_config),
            memory_service_type=agent_definition.memory_service_type,
            database_url=database_url,
            langsmith_tracing=langsmith_tracing,
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=langsmith_endpoint,
            langsmith_project=langsmith_project,
            langsmith_workspace_id=langsmith_workspace_id,
        )
    else:
        bound_services = services
    result = start_workflow(
        agent_definition.workflow_path,
        input_payload,
        run_id=run_id,
        storage_root=storage_root,
        services=bound_services,
        memory_service_type=agent_definition.memory_service_type,
        initial_state_overrides={
            "agent_id": agent_definition.agent_id,
            "agent_name": agent_definition.name,
            "agent_role": agent_definition.role,
            "agent_metadata": dict(agent_definition.metadata),
            "invocation_id": run_id or str(uuid4()),
            "runtime_profile": runtime_profile.profile_id,
            "context_policy": runtime_profile.context_policy.to_dict(),
            "memory_policy": runtime_profile.memory_policy.to_dict(),
            "allowed_tools": list(agent_definition.allowed_tools),
        },
    )
    result["agent"] = {
        "agent_id": agent_definition.agent_id,
        "name": agent_definition.name,
        "role": agent_definition.role,
        "workflow_path": agent_definition.workflow_path,
        "memory_service_type": agent_definition.memory_service_type,
        "runtime_profile": agent_definition.runtime_profile,
        "allowed_tools": list(agent_definition.allowed_tools),
    }
    return result

