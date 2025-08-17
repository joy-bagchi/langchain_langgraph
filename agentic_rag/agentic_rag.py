from typing import TypedDict, List, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import convert_to_messages
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader



# ---------- Setup (one-time) ----------
# 1) Build the vector store


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
emb = OpenAIEmbeddings(model="text-embedding-3-small")

docs_list = [item for sublist in docs for item in sublist]
print(docs_list)  # docs_list

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
print(doc_splits[0].page_content.strip())

vs = FAISS.from_documents(doc_splits, emb)
retriever = vs.as_retriever(search_kwargs={"k": 6})


response_model = init_chat_model("openai:gpt-4.1", temperature=0)

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    resp = response_model.bind_tools([retriever_tool]).invoke(state["messages"])

    return {"messages": [resp]}


# 2) LLM and prompt
llm = ChatOpenAI(model="gpt-4o-mini")  # pick what you like
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a precise, source-citing assistant. Use CONTEXT if helpful."),
    ("system", "CONTEXT:\n{context}"),
    ("human", "QUESTION:\n{question}")
])
print(PROMPT)  # PROMPT
# ---------- Graph State ----------
class RAGState(TypedDict):
    question: str
    retrieved: List[Document]
    context: str
    answer: str

# ---------- Nodes ----------
def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {**state, "retrieved": docs}

def compose_context_node(state: RAGState) -> RAGState:
    # Simple render: title + excerpt + source id. You can add line numbers/citations here.
    lines = []
    for d in state.get("retrieved", []):
        src = d.metadata.get("id", "unknown")
        lines.append(f"[{src}] {d.page_content}")
    context = "\n\n".join(lines)
    return {**state, "context": context}

def answer_node(state: RAGState) -> RAGState:
    msg = PROMPT.format_messages(question=state["question"], context=state.get("context", ""))
    resp = llm.invoke(msg)
    return {**state, "answer": resp.content}

# ---------- Wiring the graph ----------
g = StateGraph(RAGState)
g.add_node("retrieve", retrieve_node)
g.add_node("compose", compose_context_node)
g.add_node("answer", answer_node)

g.set_entry_point("retrieve")
g.add_edge("retrieve", "compose")
g.add_edge("compose", "answer")
g.add_edge("answer", END)

app = g.compile()

# Run
result = app.invoke({"question": "Where do llamas live?"})
print(result["answer"])


#%%

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model("openai:gpt-4.1", temperature=0)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
#%% md
# Run this with irrelevant documents in the tool response:

#%%

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What are the types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}
grade_documents(input)
#%% md
# Confirm that the relevant documents are classified as such:

#%%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
grade_documents(input)
#%% md
# 5. Rewrite question

# Build the rewrite_question node. The retriever tool can return potentially irrelevant documents, which indicates a
# need to improve the original user question. To do so, we will call the rewrite_question node:
#%%
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}
#%%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

response = rewrite_question(input)
print(response["messages"][-1]["content"])
#%% md
# 6. Generate an answer

# Build `generate_answer `node: if we pass the grader checks, we can generate the final answer based on the original
# question and the retrieved context:
#%%
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
#%%
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}

response = generate_answer(input)
response["messages"][-1].pretty_print()
#%% md
# 7. Assemble the graph

# - Start with a `generate_query_or_respond`and determine if we need to cal `retriever_tool`
# - Route to next step using `tools_condition`:
#     - If `generate_query_or_respond` returned `tool_calls`, call `retriever_tool` to retrieve context
#     - Otherwise, respond directly to the user
# - Grade retrieved document content for relevance to the question (`grade_documents`) and route to next step:
#     - If not relevant, rewrite the question using `rewrite_question` and then call `generate_query_or_respond` again
#     - If relevant, proceed to `generate_answer` and generate final response using the ToolMessage with the
#     retrieved document context
# API Reference: [StateGraph](https://langchain.readthedocs.io/en/latest/modules/agents/agent_types/state_graph.html#stategraph)StateGraph | START | END | ToolNode | tools_condition
#%%
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()
#%%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))