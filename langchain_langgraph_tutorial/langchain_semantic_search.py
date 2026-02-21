"""
This is a langchain_langgraph_tutorial on how to use LangChain for semantic search.
URL of the Tutorial: https://docs.langchain.com/oss/python/langchain/knowledge-base
"""

from langchain_community.document_loaders import PyPDFLoader

file_path = r"C:\Users\joyba\OneDrive - jaybagchi.com\Personal\Business Skills for Success\Change the Way You Persuade.PDF"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))