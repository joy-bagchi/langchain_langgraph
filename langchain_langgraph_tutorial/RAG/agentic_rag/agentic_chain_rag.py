"""
LangChain RAG Chains
https://docs.langchain.com/oss/python/langchain/rag#rag-chains

This module provides functionality for using RAG (Retrieval-Augmented Generation) chains in LangChain.
It includes classes and functions for creating and using RAG chains for tasks such as question answering.
"""
from langchain.agents import create_agent

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model

from semantic_search.langchain_semantic_search import vector_store

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Generate a prompt with context for the model."""
    last_query = request.state["messages"][-1].content
    retrieved_docs = vector_store.similarity_search(last_query)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    system_prompt = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{context}\n\n"
    )
    return system_prompt

agent = create_agent(
    model=init_chat_model("openai:gpt-5.1"),
    tools=[],
    middleware=[prompt_with_context],
)

query = "What is the role of Influence in Leadership?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    print(step["messages"][-1].content)
