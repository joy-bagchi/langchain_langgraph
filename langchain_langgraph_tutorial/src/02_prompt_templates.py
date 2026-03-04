"""Starter module: 02_prompt_templates.

Implement lesson logic for this module.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


def validate_non_empty_text(value: object, field_name: str) -> str:
    """Validate that a field is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")

    return cleaned


def main() -> None:
    """Run a basic langchain prompt template."""
    model = init_chat_model("openai:gpt-5.1")
    prompt_1 = ChatPromptTemplate.from_template("Tell me an interesting fact about {topic}!")
    prompt_2 = ChatPromptTemplate.from_template("What is the topic of {message}?")

    system_message = SystemMessage(content="You are a helpful mathematically inclined assistant.")
    topic = validate_non_empty_text("Singular Value Decomposition", "topic")
    response = model.invoke([system_message] + prompt_1.format_prompt(topic=topic).to_messages())
    print(f"Metadata: {response.response_metadata}")
    print(f"Content: {response.content}")

    first_response_content = validate_non_empty_text(response.content, "response.content")
    response = model.invoke([system_message] + prompt_2.format_prompt(message=first_response_content).to_messages())
    print(f"Metadata: {response.response_metadata}")
    print(f"Content: {response.content}")


if __name__ == "__main__":
    main()
