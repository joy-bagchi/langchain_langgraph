"""
This is a langchain_langgraph_tutorial on how to use LangChain for semantic search.
URL of the Tutorial: https://docs.langchain.com/oss/python/langchain/knowledge-base
"""
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

file_path = r"C:\Users\joyba\OneDrive - jaybagchi.com\Personal\Business Skills for Success\Change the Way You Persuade.PDF"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(f"Number of pages in the document: {len(docs)}")
print(f"{docs[1].page_content[:100]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"Number of chunks: {len(texts)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_1 = embeddings.embed_query(texts[0].page_content)
vector_2 = embeddings.embed_query(texts[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Length of vector: {len(vector_1)}")
print(f"Vector 1: {vector_1[:10]}")

embedding_dim = len(vector_1)
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(embedding_function=embeddings,
                     index=index,
                     docstore=InMemoryDocstore({}),
                     index_to_docstore_id={})

ids = vector_store.add_documents(documents=texts)

results = vector_store.similarity_search(
    "What are the different ways to persuade people?",
)

print(f"Query: What are the different ways to persuade people?")
print(f"Number of results: {len(results)}")
for result in results:
    print(f"Result: {result.page_content[:2000]}")
    print("-"*50)
    print()

## Retrievers
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results = retriever.batch(
    [
        "What are the different ways to persuade people?",
        "What are the Pros and Cons of each approach?",
    ],
)

print(f"Query: Retrieval results for two questions")
print(f"Number of results: {len(results)}")
for result in results:
    print(f"Result: {result}")
    print("-"*50)
    print()

__all__ = ["file_path", "loader", "docs", "text_splitter", "texts", "embeddings", "vector_store", "ids", "results", "retriever"]