Based on the documentation, I can see that LangGraph focuses on low-level orchestration and doesn't mention a prebuilt `create_react_agent` function. However, the documentation does mention that LangChain provides agent abstractions built on top of LangGraph. Let me create an example based on the patterns shown in the documentation, using a custom stock pricing tool.

Here's a short example of a LangGraph agent that can look up stock pricing information:

```python
import os
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

# Set up your API key
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Initialize the model
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)

# Define a mock stock pricing tool
@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given symbol.
    
    Args:
        symbol: The stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
    """
    # In a real implementation, you would call an actual API like Alpha Vantage, Yahoo Finance, etc.
    # This is a mock implementation for demonstration
    mock_prices = {
        "AAPL": "$150.25",
        "GOOGL": "$2,750.80",
        "TSLA": "$242.15",
        "MSFT": "$378.90",
        "AMZN": "$3,245.67"
    }
    
    symbol = symbol.upper()
    if symbol in mock_prices:
        return f"The current price of {symbol} is {mock_prices[symbol]}"
    else:
        return f"Sorry, I don't have pricing information for {symbol}. Available symbols: {', '.join(mock_prices.keys())}"

# Set up tools
tools = [get_stock_price]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Define state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Define nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful financial assistant. You can look up current stock prices for users. When a user asks about stock prices, use the get_stock_price tool to get accurate information."
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

# Define routing logic
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, we stop (reply to the user)
    return END

# Build the agent
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
stock_agent = agent_builder.compile()

# Example usage
if __name__ == "__main__":
    # Test the agent
    messages = [HumanMessage(content="What's the current price of Apple stock?")]
    result = stock_agent.invoke({"messages": messages})
    
    print("Conversation:")
    for message in result["messages"]:
        if hasattr(message, 'content'):
            role = message.__class__.__name__
            print(f"{role}: {message.content}")
```

This example creates a LangGraph agent that can look up stock pricing information. Here's what it does:

## Key Components:

1. **Stock Pricing Tool**: The `get_stock_price` tool simulates looking up stock prices. In a real implementation, you'd integrate with an actual financial API like:
   - Alpha Vantage
   - Yahoo Finance API
   - Polygon.io
   - IEX Cloud

2. **Agent State**: Uses `MessagesState` to maintain conversation history.

3. **LLM Node**: The agent uses Claude to understand user requests and decide whether to call the stock pricing tool.

4. **Tool Node**: Executes the stock price lookup when needed.

5. **Routing Logic**: Determines whether to continue with tool calls or end the conversation.

## To make this work with real stock data:

Replace the mock `get_stock_price` tool with a real API integration:

```python
import requests

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given symbol."""
    # Example with Alpha Vantage API
    api_key = "YOUR_API_KEY"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    if "Global Quote" in data:
        price = data["Global Quote"]["05. price"]
        return f"The current price of {symbol} is ${price}"
    else:
        return f"Could not retrieve price for {symbol}"
```

## Usage Examples:

```python
# Ask about a specific stock
result = stock_agent.invoke({
    "messages": [HumanMessage(content="What's Tesla's current stock price?")]
})

# Ask about multiple stocks
result = stock_agent.invoke({
    "messages": [HumanMessage(content="Can you tell me the prices for Apple and Microsoft?")]
})
```

This agent follows the React (Reasoning + Acting) pattern where the LLM reasons about what action to take and then uses tools to gather information before providing a response.