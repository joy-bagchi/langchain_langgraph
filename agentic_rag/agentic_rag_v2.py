import requests
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from markdownify import markdownify

@tool
def fetch_webpage(url: str) -> str:
    """Fetches the content of a webpage."""
    response = requests.get(url, timeout=10)
    return response.text

system_prompt = """"\
Use fetch_webpage when you need to fetch information from a web-page; quote relevant snippets."""


agent = create_agent(
    model=init_chat_model("openai:gpt-5.1"),
    tools=[fetch_webpage],
    system_prompt=system_prompt,
)

result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
print(result.get("messages"))

