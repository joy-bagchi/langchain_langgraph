"""Starter module: 01_hello_chat.

Implement lesson logic for this module.
"""

from __future__ import annotations
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


def main() -> None:
    """Run a basic langchain chat model"""
    model = init_chat_model("openai:gpt-5.1")
    system_message = SystemMessage(content="You are a mathematically inclined assistant.")
    human_message = HumanMessage(content="Hello, LangChain!")
    response = model.invoke([system_message, human_message])
    print(f"Metadata: {response.response_metadata}")
    print(f"Content: {response.content}")


if __name__ == "__main__":
    main()
