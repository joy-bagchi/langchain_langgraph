"""Starter module: prompt_lab.

Implement lesson logic for this module.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

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

    Supports:
    - `explain`: concise beginner explanation.
    - `quiz`: short quiz with answer key.
    """
    if mode == "explain":
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a concise technical assistant."),
                ("human", "Explain {topic} for a beginner in 3 bullet points."),
            ]
        )

    if mode == "quiz":
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a supportive tutor. Create short formative quizzes.",
                ),
                (
                    "human",
                    "Create a 3-question quiz about {topic} and include a brief answer key.",
                ),
            ]
        )

    raise ValueError(f"Unsupported mode: {mode}")


def build_chain(mode: str, model_name: str) -> Any:
    """Compose prompt -> model -> parser.
    """
    prompt = build_prompt(mode=mode)
    model = init_chat_model(model_name)
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain


def validate_topic(topic: str) -> str:
    """Validate topic input for CLI and programmatic use."""
    cleaned = topic.strip()
    if not cleaned:
        raise ValueError("topic must be a non-empty string.")
    return cleaned


def run_once(topic: str, mode: str, model_name: str) -> str:
    """Execute one prompt-lab run and return parsed text.
    """
    valid_topic = validate_topic(topic)
    chain = build_chain(mode=mode, model_name=model_name)
    output = chain.invoke({"topic": valid_topic})
    return cast(str, output)


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
    except (ValueError, NotImplementedError) as error:
        print(error)


if __name__ == "__main__":
    main()
