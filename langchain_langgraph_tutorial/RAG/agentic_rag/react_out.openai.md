Below is a minimal LangGraph agent using the **prebuilt `create_react_agent`** helper. It exposes one tool, `get_stock_price`, which fetches a quote (example uses the free Stooq endpoint; you can swap in any pricing API).

```python
# pip install -U langgraph semantic_search-core semantic_search-community

from typing import Optional
import csv
import urllib.request

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent


@tool
def get_stock_price(symbol: str) -> dict:
    """Look up the latest stock price for a ticker symbol (e.g., AAPL, MSFT)."""
    symbol = symbol.upper().strip()

    # Free, no-auth endpoint (delayed data): https://stooq.com/q/l/?s=aapl.us&f=sd2t2ohlcv&h&e=csv
    # For US tickers, Stooq expects ".US"
    stooq_symbol = symbol if "." in symbol else f"{symbol}.US"
    url = f"https://stooq.com/q/l/?s={stooq_symbol}&f=sd2t2ohlcv&h&e=csv"

    with urllib.request.urlopen(url, timeout=10) as resp:
        text = resp.read().decode("utf-8")

    row = next(csv.DictReader(text.splitlines()))
    if not row or row.get("Close") in (None, "", "N/A"):
        raise ValueError(f"No price found for symbol={symbol}")

    return {
        "symbol": symbol,
        "date": row["Date"],
        "time": row["Time"],
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": int(float(row["Volume"])) if row.get("Volume") else None,
        "source": "stooq",
    }


# Any chat model that supports tool calling works here.
# Example from LangGraph docs uses init_chat_model(...)
model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)

agent = create_react_agent(
    model=model,
    tools=[get_stock_price],
    prompt="You are a helpful finance assistant. Use tools to fetch prices when needed.",
)

# Invoke with a user message
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the latest close price for AAPL?"}]}
)

# Final assistant message is the last one
print(result["messages"][-1].content)
```

Notes (from LangGraph agent patterns): LangGraph agents typically run an LLM + tools loop; the prebuilt `create_react_agent` packages that loop for you, so you only provide a tool list and a model that supports tool calling.