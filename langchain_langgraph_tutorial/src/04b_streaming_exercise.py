"""Bonus exercise: model streaming with LangChain runnables.

Goal:
- Produce streamed model output.
- Print chunks as they arrive.
- Capture chunks into a final string.

How to use:
1) Read each
2) Run this module directly to test your progress.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_streaming_chain(model_name: str = "openai:gpt-5.1"):
    """Return a simple chain that supports token/text streaming."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical tutor."),
            (
                "human",
                "Teach me the core idea of {topic} in short, clear bullet points.",
            ),
        ]
    )
    model = init_chat_model(model_name)
    parser = StrOutputParser()
    return prompt | model | parser


def print_streamed_response(topic:  str) -> None:
    """Exercise 1: print model output as chunks arrive.

    - Build the chain.
    - Iterate over `chain.stream(...)`.
    - Print each chunk immediately (`end=""`, `flush=True`).
    - Print a newline at the end.
    """
    chain = build_streaming_chain()
    for chunk in chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
    print()


def capture_streamed_response(topic: str) -> str:
    """Exercise 2: capture streamed chunks into one final string.

    - Build the chain.
    - Collect chunks from `chain.stream(...)` into a list.
    - Join chunks into one string and return it.
    """
    chain = build_streaming_chain()
    chunks = []
    for chunk in chain.stream({"topic": topic}):
        chunks.append(chunk)
    return "".join(chunks)


def main() -> None:
    """Run both streaming exercises with one sample topic."""
    topic = "vector databases"

    print("=== Exercise 1: live streaming output ===")
    try:
        print_streamed_response(topic)
    except NotImplementedError as error:
        print(error)

    print("\n=== Exercise 2: captured streaming output ===")
    try:
        final_text = capture_streamed_response(topic)
        print(final_text)
    except NotImplementedError as error:
        print(error)


if __name__ == "__main__":
    main()
