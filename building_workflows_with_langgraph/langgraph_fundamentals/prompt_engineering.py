from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
prompt_template =  "Tell me a joke about {topic}."
msg_template = HumanMessagePromptTemplate.from_template(prompt_template)
msg_example = msg_template.format(topic="cats")
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    msg_template])

chain = chat_prompt_template | llm | StrOutputParser()
response = chain.invoke({"topic": "programming"})
print(response)
