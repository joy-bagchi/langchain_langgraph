"""Coding runtime service for bounded ProductDNA coding jobs.

Slice 0 deliberately uses the Codex CLI's non-interactive ``codex exec`` mode.
ProductDNA/Agentic OS owns the job boundary and human gate; Codex owns the
inner inspect -> edit -> run -> repair loop inside the supplied workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from agentic_harness.shared.services import ServiceDescriptor


@dataclass(slots=True)
class CodingJob:
    """One bounded coding slice delegated to a coding runtime."""

    job_id: str
    prompt: str
    workspace: str | Path
    sandbox: str = "workspace-write"
    model: str | None = None
    ephemeral: bool = True
    skip_git_repo_check: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CodingJobResult:
    """Machine-readable result returned to Agentic OS for the human gate."""

    job_id: str
    status: str
    return_code: int
    workspace: str
    thread_id: str | None = None
    final_message: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    stderr: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded" and self.return_code == 0


class CodingRuntimeService(Protocol):
    """Contract for a provider-specific coding-agent runtime."""

    descriptor: ServiceDescriptor

    def run(self, job: CodingJob) -> CodingJobResult:
        """Run one bounded coding job to completion."""


class CodexExecRuntime:
    """Thin adapter over ``codex exec`` for the first ProductDNA slice.

    The adapter intentionally does not recreate Codex file, shell, sandbox,
    or repair-loop functionality. It delegates those concerns to the Codex
    harness and returns structured execution evidence to Agentic OS.
    """

    def __init__(
        self,
        *,
        codex_command: str = "codex",
        base_args: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.codex_command = codex_command
        self.base_args = tuple(base_args or ())
        self.environment = dict(environment or {})
        self.descriptor = ServiceDescriptor(
            service_name="coding_runtime",
            implementation_id="codex_exec_runtime",
            maturity="experimental",
            capabilities=[
                "bounded_coding_job",
                "workspace_write",
                "command_execution",
                "jsonl_evidence",
            ],
        )

    def build_command(self, job: CodingJob) -> list[str]:
        """Build the non-interactive Codex command without embedding secrets."""
        command = [self.codex_command, "exec", "--json"]
        command.extend(self.base_args)
        command.extend(["--sandbox", job.sandbox])
        if job.ephemeral:
            command.append("--ephemeral")
        if job.model:
            command.extend(["--model", job.model])
        if job.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        command.append(job.prompt)
        return command

    def run(self, job: CodingJob) -> CodingJobResult:
        workspace = Path(job.workspace).expanduser().resolve()
        if not workspace.exists():
            raise FileNotFoundError(f"Coding workspace does not exist: {workspace}")
        if not workspace.is_dir():
            raise NotADirectoryError(f"Coding workspace is not a directory: {workspace}")

        env = os.environ.copy()
        env.update(self.environment)

        try:
            completed = subprocess.run(
                self.build_command(job),
                cwd=workspace,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Codex executable '{self.codex_command}' was not found. "
                "Install/configure the Codex CLI in the coding-worker environment."
            ) from exc

        events = self._parse_events(completed.stdout)
        thread_id = self._thread_id(events)
        final_message = self._final_message(events)
        status = self._status(events, completed.returncode)

        return CodingJobResult(
            job_id=job.job_id,
            status=status,
            return_code=completed.returncode,
            workspace=str(workspace),
            thread_id=thread_id,
            final_message=final_message,
            events=events,
            stderr=completed.stderr,
            metadata={
                "runtime": "codex_exec",
                "sandbox": job.sandbox,
                "model": job.model,
                **job.metadata,
            },
        )

    @staticmethod
    def _parse_events(stdout: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                # Preserve unexpected output as evidence instead of discarding it.
                events.append({"type": "unparsed_output", "text": payload})
                continue
            if isinstance(item, dict):
                events.append(item)
            else:
                events.append({"type": "unparsed_output", "value": item})
        return events

    @staticmethod
    def _thread_id(events: list[dict[str, Any]]) -> str | None:
        for event in events:
            if event.get("type") == "thread.started":
                value = event.get("thread_id")
                return str(value) if value else None
        return None

    @staticmethod
    def _final_message(events: list[dict[str, Any]]) -> str | None:
        final: str | None = None
        for event in events:
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict) or item.get("type") != "agent_message":
                continue
            text = item.get("text")
            if text is not None:
                final = str(text)
        return final

    @staticmethod
    def _status(events: list[dict[str, Any]], return_code: int) -> str:
        event_types = {str(event.get("type", "")) for event in events}
        if return_code == 0 and "turn.completed" in event_types:
            return "succeeded"
        if "turn.failed" in event_types or "error" in event_types:
            return "failed"
        return "succeeded" if return_code == 0 else "failed"
