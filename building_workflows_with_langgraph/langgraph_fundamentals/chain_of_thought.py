from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

llm = OpenAI()
math_cot_prompt = hub.pull("arietem/math_cot")
cot_chain = math_cot_prompt | llm | StrOutputParser()
print(cot_chain.invoke("Solve Equation 2x + 3 = 7."))