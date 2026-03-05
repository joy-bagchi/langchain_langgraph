"""Starter module: 04_chains.

Implement lesson logic for this module.
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_prompt() -> ChatPromptTemplate:
    """Return the reusable prompt for this lesson."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical assistant."),
            ("human", "Explain {topic} in 3 bullet points for a beginner."),
        ]
    )


def build_chain(model_name: str = "openai:gpt-5.1") -> Any:
    """Create and return the runnable chain.

    Day 4 objective:
    - Compose prompt -> model -> parser.
    - Return the composed runnable for reuse.
    """
    prompt = build_prompt()
    model = init_chat_model(model_name)
    parser = StrOutputParser()
    chain = prompt | model | parser
    return chain


def run_lesson_example(topic: str) -> str:
    """Invoke the chain for a single topic and return parsed output."""
    chain = build_chain()
    return chain.invoke({"topic": topic})


def main() -> None:
    """Run a tiny smoke example for manual testing."""
    topic = "vector databases"
    try:
        result = run_lesson_example(topic)
        print(result)
    except NotImplementedError as error:
        print(error)


if __name__ == "__main__":
    main()
