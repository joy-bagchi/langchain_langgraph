from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# This example demonstrates how to create a complex chain of LLM interactions using LangChain.
llm = ChatOpenAI(model="gpt-4o")

# The first chain generates the story
story_prompt = PromptTemplate.from_template("Write a short story about {topic}.")
story_chain = story_prompt | llm | StrOutputParser()
# The second chain generates a summary of the story
summary_prompt = PromptTemplate.from_template("Summarize the following story: {story}.")
summary_chain = summary_prompt | llm | StrOutputParser()

# Composing the two chains into a complex chain
complex_chain = story_chain | summary_chain
context_preserving_chain = RunnablePassthrough.assign(story=story_chain).assign(summary=summary_chain)
story_summary_with_context = context_preserving_chain.invoke({"topic": "programming"})
print(story_summary_with_context.keys())
print("Story:")
print(story_summary_with_context['story'])

print("Summary:")
print(story_summary_with_context['summary'])