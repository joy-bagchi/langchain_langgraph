"""CLI entrypoint for the memory-aware workflow runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_harness.llm import build_model_callable, resolve_llm_config
from agentic_harness.markdown_workflow import load_workflow_definition
from agentic_harness.outputs import select_output
from agentic_harness.agentic_os import resume_declarative_workflow, run_declarative_workflow
from agentic_harness.runtime import inspect_run, resume_workflow, run_agent_workflow, start_workflow


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_agent_input_payload(*, input_path: str | None, query: str | None) -> dict:
    payload = _load_json(input_path)
    if query:
        payload["query"] = query
    return payload


def _build_generic_input_payload(*, input_path: str | None, query: str | None) -> dict:
    payload = _load_json(input_path)
    if query:
        payload["query"] = query
    return payload


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-mode",
        choices=["internal", "artifact", "response"],
        default="response",
        help="Choose whether to print internal run state, a public artifact, or an audience-formatted response.",
    )
    parser.add_argument(
        "--audience",
        choices=["human", "agent"],
        default="human",
        help="Audience hint used when output-mode is response.",
    )
    parser.add_argument(
        "--response-format",
        choices=["auto", "json", "text"],
        default="auto",
        help="Response format used when output-mode is response.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Run structured markdown workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new workflow run.")
    _add_output_arguments(run_parser)
    run_parser.add_argument("--workflow", required=True, help="Path to the workflow markdown file.")
    run_parser.add_argument("--input", help="Path to a JSON file containing input payload.")
    run_parser.add_argument("--run-id", help="Optional explicit run id.")
    run_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    run_parser.add_argument("--llm-provider", choices=["none", "openai"], help="Prompt-step LLM provider.")
    run_parser.add_argument("--model", help="Model name for prompt-step execution.")
    run_parser.add_argument("--temperature", type=float, help="Sampling temperature for prompt-step execution.")

    agent_parser = subparsers.add_parser("run-agent", help="Start a new agent-bound workflow run.")
    _add_output_arguments(agent_parser)
    agent_parser.add_argument("--agent", required=True, help="Path to the agent YAML definition.")
    agent_parser.add_argument("--input", help="Path to a JSON file containing input payload.")
    agent_parser.add_argument("--query", help="Shortcut query string for agents that expect a top-level query input.")
    agent_parser.add_argument("--run-id", help="Optional explicit run id.")
    agent_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")

    dag_parser = subparsers.add_parser("run-dag", help="Execute a declarative DAG workflow.")
    _add_output_arguments(dag_parser)
    dag_parser.add_argument("--workflow", required=True, help="Path to the declarative workflow YAML file.")
    dag_parser.add_argument("--input", help="Path to a JSON file containing workflow input payload.")
    dag_parser.add_argument("--query", help="Shortcut query string for workflows that expect a top-level query input.")
    dag_parser.add_argument("--run-id", help="Optional explicit run id.")
    dag_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    dag_parser.add_argument(
        "--auto-approve-gates",
        action="store_true",
        help="Automatically complete human gate nodes instead of pausing.",
    )

    dag_resume_parser = subparsers.add_parser("resume-dag", help="Resume a declarative DAG workflow run.")
    _add_output_arguments(dag_resume_parser)
    dag_resume_parser.add_argument("--run-id", required=True, help="DAG run id to resume.")
    dag_resume_parser.add_argument(
        "--decision",
        choices=["approved", "rejected"],
        default="approved",
        help="Decision for the pending human gate.",
    )
    dag_resume_parser.add_argument("--notes", help="Optional human gate review notes.")
    dag_resume_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    dag_resume_parser.add_argument(
        "--auto-approve-gates",
        action="store_true",
        help="Automatically complete any subsequent human gates after resuming.",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume a saved workflow run.")
    _add_output_arguments(resume_parser)
    resume_parser.add_argument("--run-id", required=True, help="Run id to resume.")
    resume_parser.add_argument("--decision", choices=["approved", "rejected"], help="Review decision for a pending review step.")
    resume_parser.add_argument("--notes", help="Optional review notes.")
    resume_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    resume_parser.add_argument("--llm-provider", choices=["none", "openai"], help="Prompt-step LLM provider.")
    resume_parser.add_argument("--model", help="Model name for prompt-step execution.")
    resume_parser.add_argument("--temperature", type=float, help="Sampling temperature for prompt-step execution.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved workflow run.")
    _add_output_arguments(inspect_parser)
    inspect_parser.add_argument("--run-id", required=True, help="Run id to inspect.")
    inspect_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    return parser


def main() -> None:
    """Dispatch CLI subcommands."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        definition = load_workflow_definition(args.workflow)
        llm_config = resolve_llm_config(
            workflow_definition=definition,
            provider=args.llm_provider,
            model=args.model,
            temperature=args.temperature,
        )
        result = start_workflow(
            args.workflow,
            _load_json(args.input),
            run_id=args.run_id,
            storage_root=args.storage_root,
            model_callable=build_model_callable(llm_config),
        )
    elif args.command == "run-agent":
        result = run_agent_workflow(
            args.agent,
            _build_agent_input_payload(input_path=args.input, query=args.query),
            run_id=args.run_id,
            storage_root=args.storage_root,
        )
    elif args.command == "run-dag":
        result = run_declarative_workflow(
            args.workflow,
            _build_generic_input_payload(input_path=args.input, query=args.query),
            run_id=args.run_id,
            storage_root=args.storage_root,
            auto_approve_human_gates=args.auto_approve_gates,
        )
    elif args.command == "resume-dag":
        result = resume_declarative_workflow(
            args.run_id,
            storage_root=args.storage_root,
            decision=args.decision,
            notes=args.notes,
            auto_approve_human_gates=args.auto_approve_gates,
        )
    elif args.command == "resume":
        run_state = inspect_run(args.run_id, storage_root=args.storage_root)
        definition = load_workflow_definition(run_state["workflow_path"])
        llm_config = resolve_llm_config(
            workflow_definition=definition,
            provider=args.llm_provider,
            model=args.model,
            temperature=args.temperature,
        )
        result = resume_workflow(
            args.run_id,
            storage_root=args.storage_root,
            decision=args.decision,
            notes=args.notes,
            model_callable=build_model_callable(llm_config),
        )
    else:
        result = inspect_run(args.run_id, storage_root=args.storage_root)

    output = select_output(
        result,
        output_mode=args.output_mode,
        audience=args.audience,
        response_format=args.response_format,
    )
    if (
        args.output_mode == "response"
        and isinstance(output, dict)
        and output.get("response_format") == "text"
        and isinstance(output.get("content"), str)
    ):
        print(output["content"])
    else:
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

