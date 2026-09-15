import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_harness.agentic_os.coding_runtime_service import CodingJob, CodexExecRuntime


def test_build_command_is_bounded_and_non_interactive(tmp_path: Path):
    runtime = CodexExecRuntime()
    job = CodingJob(
        job_id="slice-0",
        prompt="Create hello.py and run it.",
        workspace=tmp_path,
        model="gpt-5.3-codex",
    )

    command = runtime.build_command(job)

    assert command[:3] == ["codex", "exec", "--json"]
    assert ["--sandbox", "workspace-write"] == command[3:5]
    assert "--ephemeral" in command
    assert command[-1] == "Create hello.py and run it."


def test_run_returns_structured_evidence_for_human_gate(tmp_path: Path):
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item-1",
                        "type": "agent_message",
                        "text": "Created hello.py and verified it prints hello.",
                    },
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}}),
        ]
    )
    completed = subprocess.CompletedProcess(
        args=["codex"], returncode=0, stdout=stdout, stderr=""
    )
    runtime = CodexExecRuntime()
    job = CodingJob(job_id="slice-0", prompt="Do the slice", workspace=tmp_path)

    with patch("subprocess.run", return_value=completed) as run:
        result = runtime.run(job)

    assert result.succeeded
    assert result.thread_id == "thread-123"
    assert result.final_message == "Created hello.py and verified it prints hello."
    assert result.metadata["runtime"] == "codex_exec"
    assert run.call_args.kwargs["cwd"] == tmp_path.resolve()


def test_run_preserves_failure_for_human_review(tmp_path: Path):
    stdout = json.dumps(
        {"type": "turn.failed", "error": {"message": "test command failed"}}
    )
    completed = subprocess.CompletedProcess(
        args=["codex"], returncode=1, stdout=stdout, stderr="failure"
    )
    runtime = CodexExecRuntime()

    with patch("subprocess.run", return_value=completed):
        result = runtime.run(
            CodingJob(job_id="slice-0", prompt="Do the slice", workspace=tmp_path)
        )

    assert not result.succeeded
    assert result.status == "failed"
    assert result.stderr == "failure"


def test_missing_workspace_fails_before_invoking_codex(tmp_path: Path):
    runtime = CodexExecRuntime()

    with pytest.raises(FileNotFoundError):
        runtime.run(
            CodingJob(
                job_id="slice-0",
                prompt="Do the slice",
                workspace=tmp_path / "missing",
            )
        )
