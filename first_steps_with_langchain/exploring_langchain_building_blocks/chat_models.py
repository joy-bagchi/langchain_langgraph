from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_model = ChatOpenAI(model="gpt-4o")
response = chat_model.invoke(
    [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Write a python function to calculate factorial"),
    ]
)

print(response)
