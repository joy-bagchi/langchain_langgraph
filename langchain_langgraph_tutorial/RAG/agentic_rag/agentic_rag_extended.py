import requests
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from markdownify import markdownify


ALLOWED_DOMAINS = ["https://langchain-ai.github.io/", "https://docs.langchain.com/"]
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt'


@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            f"Error: URL {url} not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)


# We will fetch the content of llms.txt (a list of approved doc URLs), so this can
# be done ahead of time without requiring an LLM request. This is not the docs
# themselves; it is a source allow-list used by the agent during tool-based retrieval.
llms_txt_content = requests.get(LLMS_TXT).text

# System prompt for the agent
system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about LangGraph and related tools.

Instructions:

1. If a user asks a question you're unsure about — or one that likely involves API usage,
   behavior, or configuration — you MUST use the `fetch_documentation` tool to consult the relevant docs.
2. When citing documentation, summarize clearly and include relevant context from the content.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best expert understanding.

You can access official documentation from the following approved URL list.
This list is not the documentation itself. The actual RAG step happens when
you call `fetch_documentation`, which retrieves content and injects it into
the conversation context before generating an answer.

{llms_txt_content}

You MUST consult the documentation to get up to date documentation
before answering a user's question about LangGraph.

Your answers should be clear, concise, and technically accurate.
"""

tools = [fetch_documentation]

# model = init_chat_model("claude-sonnet-4-0", max_tokens=32_000)
model = init_chat_model("openai:gpt-5.2", max_tokens=32_000)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic_RAG",
)

response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent using the "
            "prebuilt create agent. The agent should be able "
            "to look up stock pricing information."
        ))
    ]
})

print(response['messages'][-1].content)
