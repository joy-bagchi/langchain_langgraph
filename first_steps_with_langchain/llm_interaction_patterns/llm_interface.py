from langchain_openai import OpenAI

openai_llm = OpenAI()
response = openai_llm.invoke("Say hello to the world and introduce yourself.")
print(response)
