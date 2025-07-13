from langchain_community.llms import FakeListLLM

fakellm = FakeListLLM(
    responses=[
        "Hello, world! I am a simulated LLM, here to assist you with your queries.",
        "I can help you with various tasks, from answering questions to providing information.",
    ]
)

response = fakellm.invoke ("What is the meaning of life?")
print(response)
