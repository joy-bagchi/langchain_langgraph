"""Starter module: 03_output_parsing.

Implement lesson logic for this module.
"""

from __future__ import annotations

from typing import Any
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    content: str = Field(description="The content of the response")
    confidence: float = Field(description="The confidence of the answer")


def main() -> None:
    """Run a minimal placeholder entrypoint."""
    model = init_chat_model("openai:gpt-5.1")
    structured_model = model.with_structured_output(ModelResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful mathematically inclined assistant."),
        ("human", "Please provide an interesting fact about {topic}!"),
    ])
    chain = prompt | structured_model
    try:
        response = chain.invoke({"topic": "Singular Value Decomposition"})
        print(f"Answer: {response.answer}")
        print(f"Content: {response.content}")
        print(f"Confidence: {response.confidence}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
