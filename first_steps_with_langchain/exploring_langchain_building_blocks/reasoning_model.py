from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can reason about the world."),
        ("human", "{input}"),
    ]
)

chat_model = ChatOpenAI(model="o3-mini", reasoning_effort="high")

chat = template | chat_model
response = chat.invoke({"input": "What is a change of measure in Quantitative Finance?"})
print(response.content)
