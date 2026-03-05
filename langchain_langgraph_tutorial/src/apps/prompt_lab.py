"""Starter module: prompt_lab.

Implement lesson logic for this module.
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the prompt lab app."""
    parser = argparse.ArgumentParser(
        description="Week 1 Day 5 prompt lab (CLI-style app)."
    )
    parser.add_argument(
        "--topic",
        default="LangChain",
        help="Topic to explain (default: LangChain).",
    )
    parser.add_argument(
        "--mode",
        default="explain",
        choices=["explain", "quiz"],
        help="Prompt mode to run.",
    )
    parser.add_argument(
        "--model",
        default="openai:gpt-5.1",
        help="Model name in provider:model format.",
    )
    return parser


def build_prompt(mode: str) -> ChatPromptTemplate:
    """Return prompt template based on selected mode.

    TODO:
    - Support `explain` and `quiz` with different instructions.
    - Keep one shared variable: `{topic}`.
    """
    raise NotImplementedError("TODO: build prompt templates for both modes.")


def build_chain(mode: str, model_name: str) -> Any:
    """Compose prompt -> model -> parser.

    TODO:
    - Reuse `build_prompt(mode)`.
    - Initialize chat model from `model_name`.
    - Return composed runnable chain.
    """
    _ = StrOutputParser()
    _ = init_chat_model(model_name)
    raise NotImplementedError("TODO: compose and return chain.")


def run_once(topic: str, mode: str, model_name: str) -> str:
    """Execute one prompt-lab run and return parsed text.

    TODO:
    - Invoke chain with `{'topic': topic}`.
    - Return final string output.
    """
    _ = build_chain(mode=mode, model_name=model_name)
    raise NotImplementedError("TODO: call chain.invoke and return output.")


def main() -> None:
    """Run CLI entrypoint for prompt lab.

    A CLI-style app means running the script with command-line flags, e.g.:
    - `python src/apps/prompt_lab.py --topic "vector db" --mode explain`
    """
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_once(topic=args.topic, mode=args.mode, model_name=args.model)
        print(result)
    except NotImplementedError as error:
        print(error)


if __name__ == "__main__":
    main()
