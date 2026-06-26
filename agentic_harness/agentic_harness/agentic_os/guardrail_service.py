"""Guardrail service contract and rule-based default implementation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_harness.contracts import WorkflowGraphState, WorkflowStep
from agentic_harness.shared.services import GuardrailDecision, ServiceDescriptor


@dataclass(slots=True)
class GuardrailRequest:
    phase: str
    step: WorkflowStep
    state: WorkflowGraphState
    candidate_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GuardrailService(Protocol):
    descriptor: ServiceDescriptor

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate pre-step or post-step guardrails."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _looks_like_credit_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = list(map(int, reversed(digits)))
    for index, digit in enumerate(reverse_digits):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class PassthroughGuardrailService:
    """Simple allow-all guardrail implementation."""

    def __init__(self) -> None:
        self.descriptor = ServiceDescriptor(
            service_name="guardrails",
            implementation_id="passthrough_guardrail_service",
            maturity="simple",
            capabilities=["pre_step", "post_step"],
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return GuardrailDecision(allowed=True, action="allow")


class RuleBasedGuardrailService:
    """Deterministic guardrail engine with block and escalate actions."""

    _SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
        ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    )
    _PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    )

    def __init__(self) -> None:
        self.default_policy = {
            "pre": {
                "enabled": True,
                "max_input_chars": 12000,
                "blocked_tool_ids": [],
                "block_patterns": [],
            },
            "post": {
                "enabled": True,
                "max_output_chars": 16000,
                "block_patterns": [],
                "escalate_patterns": [],
                "detect_secrets": True,
                "detect_pii": True,
            },
        }
        self.descriptor = ServiceDescriptor(
            service_name="guardrails",
            implementation_id="rule_based_guardrail_service",
            maturity="advanced",
            capabilities=[
                "pre_step",
                "post_step",
                "block",
                "escalate",
                "step_metadata_policy",
                "agent_metadata_policy",
                "secret_detection",
                "pii_detection",
            ],
        )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        policy = self._resolve_policy(request)
        phase_key = "pre" if request.phase == "pre_step" else "post"
        phase_policy = policy.get(phase_key, {})
        if not phase_policy or not phase_policy.get("enabled", True):
            return GuardrailDecision(allowed=True, action="allow")
        if request.phase == "pre_step":
            return self._evaluate_pre(request, phase_policy)
        return self._evaluate_post(request, phase_policy)

    def _resolve_policy(self, request: GuardrailRequest) -> dict[str, Any]:
        policy = _deep_merge(self.default_policy, {})
        agent_policy = (
            dict(request.state.get("agent_metadata", {})).get("guardrails", {})
            if isinstance(request.state.get("agent_metadata", {}), dict)
            else {}
        )
        step_policy = request.step.metadata.get("guardrails", {})
        if isinstance(agent_policy, dict):
            policy = _deep_merge(policy, agent_policy)
        if isinstance(step_policy, dict):
            policy = _deep_merge(policy, step_policy)
        return policy

    def _evaluate_pre(self, request: GuardrailRequest, policy: dict[str, Any]) -> GuardrailDecision:
        reasons: list[str] = []
        input_text = str(request.metadata.get("input_text", ""))
        tool_id = str(request.metadata.get("tool_id", "")).strip()
        blocked_tool_ids = {str(item).strip() for item in policy.get("blocked_tool_ids", []) if str(item).strip()}
        if tool_id and tool_id in blocked_tool_ids:
            reasons.append(f"tool '{tool_id}' is blocked by guardrail policy")
        max_input_chars = int(policy.get("max_input_chars", 0) or 0)
        if max_input_chars and len(input_text) > max_input_chars:
            reasons.append(f"rendered input exceeded guardrail limit of {max_input_chars} chars")
        for pattern in policy.get("block_patterns", []):
            regex = re.compile(str(pattern), re.IGNORECASE)
            if regex.search(input_text):
                reasons.append(f"rendered input matched blocked pattern: {pattern}")
        if reasons:
            return GuardrailDecision(
                allowed=False,
                action="block",
                reasons=reasons,
                metadata={"phase": "pre_step"},
            )
        return GuardrailDecision(allowed=True, action="allow")

    def _evaluate_post(self, request: GuardrailRequest, policy: dict[str, Any]) -> GuardrailDecision:
        reasons: list[str] = []
        escalate_reasons: list[str] = []
        output_text = self._stringify_candidate(request.candidate_output)
        max_output_chars = int(policy.get("max_output_chars", 0) or 0)
        if max_output_chars and len(output_text) > max_output_chars:
            reasons.append(f"candidate output exceeded guardrail limit of {max_output_chars} chars")
        for pattern in policy.get("block_patterns", []):
            regex = re.compile(str(pattern), re.IGNORECASE)
            if regex.search(output_text):
                reasons.append(f"candidate output matched blocked pattern: {pattern}")
        if policy.get("detect_secrets", True):
            for label, pattern in self._SECRET_PATTERNS:
                if pattern.search(output_text):
                    reasons.append(f"candidate output appears to contain a secret: {label}")
        for pattern in policy.get("escalate_patterns", []):
            regex = re.compile(str(pattern), re.IGNORECASE)
            if regex.search(output_text):
                escalate_reasons.append(f"candidate output matched escalation pattern: {pattern}")
        if policy.get("detect_pii", True):
            for label, pattern in self._PII_PATTERNS:
                if pattern.search(output_text):
                    escalate_reasons.append(f"candidate output may contain sensitive data: {label}")
            if _looks_like_credit_card(output_text):
                escalate_reasons.append("candidate output may contain a payment card number")
        if reasons:
            return GuardrailDecision(
                allowed=False,
                action="block",
                reasons=reasons,
                metadata={"phase": "post_step"},
            )
        if escalate_reasons:
            return GuardrailDecision(
                allowed=False,
                action="escalate",
                reasons=escalate_reasons,
                metadata={"phase": "post_step"},
            )
        return GuardrailDecision(allowed=True, action="allow")

    @staticmethod
    def _stringify_candidate(candidate_output: Any) -> str:
        if candidate_output is None:
            return ""
        if isinstance(candidate_output, str):
            return candidate_output
        try:
            return json.dumps(candidate_output, sort_keys=True)
        except TypeError:
            return str(candidate_output)

