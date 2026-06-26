"""Evaluation service contract and rule-based default implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_harness.shared.services import ServiceDescriptor


@dataclass(slots=True)
class EvaluationRequest:
    phase: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResponse:
    status: str = "skipped"
    score: float | None = None
    findings: list[str] = field(default_factory=list)
    action: str = "allow"
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationService(Protocol):
    descriptor: ServiceDescriptor

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Evaluate runtime behavior or artifacts."""


class BasicEvaluationService:
    """Rule-based runtime evaluator that can influence step routing."""

    def __init__(self) -> None:
        self.default_policy = {
            "enabled": True,
            "min_output_chars": 0,
            "required_patterns": [],
            "banned_patterns": [],
            "escalate_patterns": [],
            "retry_on_empty_output": False,
            "critic": {
                "enabled": False,
                "required_terms": [],
                "preferred_terms": [],
                "required_step_ids": [],
                "required_output_keys": [],
                "min_score": 0.7,
                "on_below_threshold": "escalate",
            },
        }
        self.descriptor = ServiceDescriptor(
            service_name="evaluation",
            implementation_id="basic_evaluation_service",
            maturity="advanced",
            capabilities=["step_runtime_eval", "rule_based_findings", "retry", "escalate", "fail", "critic_scoring", "workflow_context_eval"],
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        policy = self._resolve_policy(request)
        if not policy.get("enabled", True):
            return EvaluationResponse(status="skipped", score=None, action="allow")

        findings: list[str] = []
        status = "passed"
        score: float | None = 1.0
        action = "allow"
        request_status = str(request.payload.get("status", "")).strip().lower()
        if request_status in {"failed", "error", "rejected"}:
            status = "failed"
            score = 0.0
            action = "fail"
            findings.append(f"{request.phase} execution reported status '{request_status}'.")
        elif request_status in {"awaiting_review", "retrying"}:
            status = "attention_required"
            score = 0.5
            findings.append(f"{request.phase} execution requires follow-up: '{request_status}'.")
        output_text = str(request.payload.get("output_text", "") or "")
        critic_summary: dict[str, Any] | None = None
        if action == "allow":
            if policy.get("retry_on_empty_output") and not output_text.strip():
                status = "attention_required"
                score = 0.2
                action = "retry"
                findings.append("candidate output was empty")
            min_output_chars = int(policy.get("min_output_chars", 0) or 0)
            if action == "allow" and min_output_chars and len(output_text.strip()) < min_output_chars:
                status = "attention_required"
                score = 0.3
                action = "retry"
                findings.append(f"candidate output shorter than {min_output_chars} chars")
            for pattern in policy.get("banned_patterns", []):
                if re.search(str(pattern), output_text, flags=re.IGNORECASE):
                    status = "failed"
                    score = 0.0
                    action = "fail"
                    findings.append(f"candidate output matched banned pattern: {pattern}")
                    break
            if action == "allow":
                missing = [
                    str(pattern)
                    for pattern in policy.get("required_patterns", [])
                    if not re.search(str(pattern), output_text, flags=re.IGNORECASE)
                ]
                if missing:
                    status = "attention_required"
                    score = 0.4
                    action = "escalate"
                    findings.extend(f"candidate output missing required pattern: {pattern}" for pattern in missing)
            if action == "allow":
                for pattern in policy.get("escalate_patterns", []):
                    if re.search(str(pattern), output_text, flags=re.IGNORECASE):
                        status = "attention_required"
                        score = 0.5
                        action = "escalate"
                        findings.append(f"candidate output matched escalation pattern: {pattern}")
                        break
        critic_policy = policy.get("critic", {})
        if action == "allow" and isinstance(critic_policy, dict) and critic_policy.get("enabled"):
            critic_status, critic_score, critic_action, critic_findings, critic_summary = self._evaluate_critic(
                output_text=output_text,
                payload=request.payload,
                policy=critic_policy,
            )
            status = critic_status
            score = critic_score
            action = critic_action
            findings.extend(critic_findings)
        return EvaluationResponse(
            status=status,
            score=score,
            findings=findings,
            action=action,
            metadata={"phase": request.phase, "policy": policy, "critic": critic_summary},
        )

    def _resolve_policy(self, request: EvaluationRequest) -> dict[str, Any]:
        policy = dict(self.default_policy)
        workflow_policy = request.payload.get("workflow_metadata", {}).get("evaluation", {})
        agent_policy = request.payload.get("agent_metadata", {}).get("evaluation", {})
        step_policy = request.payload.get("step_metadata", {}).get("evaluation", {})
        for source in (workflow_policy, agent_policy, step_policy):
            if isinstance(source, dict):
                for key, value in source.items():
                    if key == "critic" and isinstance(value, dict):
                        policy["critic"] = {**policy.get("critic", {}), **value}
                    else:
                        policy[key] = value
        return policy

    def _evaluate_critic(
        self,
        *,
        output_text: str,
        payload: dict[str, Any],
        policy: dict[str, Any],
    ) -> tuple[str, float, str, list[str], dict[str, Any]]:
        findings: list[str] = []
        output_lower = output_text.lower()
        required_terms = [str(item) for item in policy.get("required_terms", []) if str(item).strip()]
        preferred_terms = [str(item) for item in policy.get("preferred_terms", []) if str(item).strip()]
        required_step_ids = [str(item) for item in policy.get("required_step_ids", []) if str(item).strip()]
        required_output_keys = [str(item) for item in policy.get("required_output_keys", []) if str(item).strip()]

        completed_step_ids = {
            str(entry.get("step_id"))
            for entry in payload.get("step_history", [])
            if isinstance(entry, dict) and entry.get("step_id")
        }
        available_output_keys = {
            str(key)
            for key in dict(payload.get("named_outputs", {})).keys()
        }

        matched_required_terms = [term for term in required_terms if term.lower() in output_lower]
        matched_preferred_terms = [term for term in preferred_terms if term.lower() in output_lower]
        missing_required_terms = [term for term in required_terms if term not in matched_required_terms]
        missing_step_ids = [step_id for step_id in required_step_ids if step_id not in completed_step_ids]
        missing_output_keys = [key for key in required_output_keys if key not in available_output_keys]

        components: list[float] = []
        if required_terms:
            components.append(len(matched_required_terms) / len(required_terms))
        if preferred_terms:
            components.append(len(matched_preferred_terms) / len(preferred_terms))
        if required_step_ids:
            components.append((len(required_step_ids) - len(missing_step_ids)) / len(required_step_ids))
        if required_output_keys:
            components.append((len(required_output_keys) - len(missing_output_keys)) / len(required_output_keys))
        score = sum(components) / len(components) if components else 1.0

        if missing_required_terms:
            findings.extend(f"critic missing required term: {term}" for term in missing_required_terms)
        if missing_step_ids:
            findings.extend(f"critic missing required prior step: {step_id}" for step_id in missing_step_ids)
        if missing_output_keys:
            findings.extend(f"critic missing required output key: {key}" for key in missing_output_keys)

        min_score = float(policy.get("min_score", 0.7))
        if score < min_score:
            findings.append(f"critic score {score:.2f} below threshold {min_score:.2f}")

        summary = {
            "score": score,
            "required_terms": required_terms,
            "matched_required_terms": matched_required_terms,
            "preferred_terms": preferred_terms,
            "matched_preferred_terms": matched_preferred_terms,
            "required_step_ids": required_step_ids,
            "missing_step_ids": missing_step_ids,
            "required_output_keys": required_output_keys,
            "missing_output_keys": missing_output_keys,
        }

        if findings:
            action = str(policy.get("on_below_threshold", "escalate")).strip().lower() or "escalate"
            if action not in {"retry", "escalate", "fail"}:
                action = "escalate"
            status = "failed" if action == "fail" else "attention_required"
            return status, score, action, findings, summary
        return "passed", score, "allow", [], summary

