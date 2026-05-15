"""CLI entrypoint for the memory-aware workflow runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_workflow.llm import build_model_callable, resolve_llm_config
from agentic_workflow.markdown_workflow import load_workflow_definition
from agentic_workflow.runtime import inspect_run, resume_workflow, start_workflow


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Run structured markdown workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a new workflow run.")
    run_parser.add_argument("--workflow", required=True, help="Path to the workflow markdown file.")
    run_parser.add_argument("--input", help="Path to a JSON file containing input payload.")
    run_parser.add_argument("--run-id", help="Optional explicit run id.")
    run_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    run_parser.add_argument("--llm-provider", choices=["none", "openai"], help="Prompt-step LLM provider.")
    run_parser.add_argument("--model", help="Model name for prompt-step execution.")
    run_parser.add_argument("--temperature", type=float, help="Sampling temperature for prompt-step execution.")

    resume_parser = subparsers.add_parser("resume", help="Resume a saved workflow run.")
    resume_parser.add_argument("--run-id", required=True, help="Run id to resume.")
    resume_parser.add_argument("--decision", choices=["approved", "rejected"], help="Review decision for a pending review step.")
    resume_parser.add_argument("--notes", help="Optional review notes.")
    resume_parser.add_argument("--storage-root", help="Override the default .workflow_memory directory.")
    resume_parser.add_argument("--llm-provider", choices=["none", "openai"], help="Prompt-step LLM provider.")
    resume_parser.add_argument("--model", help="Model name for prompt-step execution.")
    resume_parser.add_argument("--temperature", type=float, help="Sampling temperature for prompt-step execution.")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved workflow run.")
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

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
