# ============================================================
# LANGCHAIN COMPLETE REVISION PROJECT
# Covers: Models → Prompts → Structured Output → Output Parsers
#         → Chains → Runnables → Document Loaders → Text Splitters
#         → Vector Store → Retrievers → Tools → Tool Calling
# Model: Google Gemini (gemini-1.5-pro + gemini-embedding-001)
# ============================================================

# ── SETUP ────────────────────────────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, Optional
import requests, json


# ════════════════════════════════════════════════════════════
# SECTION 1 — MODELS
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 1 — MODELS")
print("="*60)

# Chat Model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = chat_model.invoke("What is LangChain in one sentence?")
print("\n[Chat Model Output]")
print(result.content)

# Embedding Model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector = embedding_model.embed_query("LangChain is a framework for LLMs")
print(f"\n[Embedding Output] Vector length: {len(vector)}, First 3 values: {vector[:3]}")


# ════════════════════════════════════════════════════════════
# SECTION 2 — PROMPTS
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 2 — PROMPTS")
print("="*60)

# ChatPromptTemplate with system + human
prompt = ChatPromptTemplate([
    ("system", "You are an expert in {domain}. Answer concisely."),
    ("human", "{question}")
])

chain = prompt | chat_model | StrOutputParser()
result = chain.invoke({"domain": "cricket", "topic": "IPL", "question": "Who has scored the most runs in IPL?"})
print("\n[ChatPromptTemplate Output]")
print(result)

# MessagesPlaceholder for chat history
history_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
]) 

chat_history = [
    HumanMessage(content="My name is Rahul"),
    AIMessage(content="Nice to meet you Rahul!")
]

history_chain = history_prompt | chat_model | StrOutputParser()
result = history_chain.invoke({
    "chat_history": chat_history,
    "question": "Do you remember my name?"
})
print("\n[MessagesPlaceholder Output]")
print(result)


# ════════════════════════════════════════════════════════════
# SECTION 3 — STRUCTURED OUTPUT
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 3 — STRUCTURED OUTPUT")
print("="*60)

class MovieReview(BaseModel):
    title: str = Field(description="Name of the movie")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the review")
    rating: int = Field(gt=0, lt=11, description="Rating out of 10")
    pros: list[str] = Field(description="List of positive aspects")
    cons: Optional[list[str]] = Field(default=None, description="List of negative aspects if any")
    summary: str = Field(description="One line summary of the review")

structured_model = chat_model.with_structured_output(MovieReview)

review_text = """
Interstellar is a visual masterpiece. The practical effects and cinematography are stunning.
The emotional depth of the story is incredible and Hans Zimmer's score is legendary.
However, the third act gets a bit confusing and hard to follow. Overall a must watch.
"""

result = structured_model.invoke(review_text)
print("\n[Structured Output]")
print(f"Title: {result.title}")
print(f"Sentiment: {result.sentiment}")
print(f"Rating: {result.rating}/10")
print(f"Pros: {result.pros}")
print(f"Cons: {result.cons}")
print(f"Summary: {result.summary}")


# ════════════════════════════════════════════════════════════
# SECTION 4 — OUTPUT PARSERS
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 4 — OUTPUT PARSERS")
print("="*60)

# PydanticOutputParser
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=0, description="Age of the person")
    city: str = Field(description="City the person lives in")

parser = PydanticOutputParser(pydantic_object=Person)

parser_prompt = PromptTemplate(
    template="Generate a fictional Indian person's details.\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

parser_chain = parser_prompt | chat_model | parser
result = parser_chain.invoke({})
print("\n[PydanticOutputParser Output]")
print(f"Name: {result.name}, Age: {result.age}, City: {result.city}")


# ════════════════════════════════════════════════════════════
# SECTION 5 — CHAINS + RUNNABLES
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 5 — CHAINS + RUNNABLES")
print("="*60)

# Sequential Chain
template1 = PromptTemplate(
    template="Write a short 3 line poem about {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Translate this poem to Hindi:\n{poem}",
    input_variables=["poem"]
)

sequential_chain = template1 | chat_model | StrOutputParser() | template2 | chat_model | StrOutputParser()
result = sequential_chain.invoke({"topic": "monsoon"})
print("\n[Sequential Chain Output — Poem translated to Hindi]")
print(result)

# RunnableParallel + RunnablePassthrough + RunnableLambda
joke_prompt = PromptTemplate(
    template="Tell me a short joke about {topic}",
    input_variables=["topic"]
)
joke_chain = joke_prompt | chat_model | StrOutputParser()

def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

full_chain = joke_chain | parallel_chain
result = full_chain.invoke({"topic": "programmers"})
print("\n[Parallel + Passthrough + Lambda Output]")
print(f"Joke: {result['joke']}")
print(f"Word count: {result['word_count']}")


# ════════════════════════════════════════════════════════════
# SECTION 6 — DOCUMENT LOADERS + TEXT SPLITTERS
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 6 — DOCUMENT LOADERS + TEXT SPLITTERS")
print("="*60)

# Using raw documents instead of a PDF file for portability
raw_docs = [
    Document(page_content="LangChain is a framework designed to simplify the creation of applications using large language models. It provides tools for chaining components together in powerful ways.", metadata={"source": "intro.txt"}),
    Document(page_content="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses them as context for the language model to generate accurate answers.", metadata={"source": "rag.txt"}),
    Document(page_content="Vector stores are databases optimised for storing and searching embeddings. Chroma and FAISS are two popular options. They allow semantic similarity search across thousands of documents efficiently.", metadata={"source": "vectorstores.txt"}),
    Document(page_content="Agents in LangChain use tools to take actions in the world. An agent decides which tool to call, runs it, and uses the result to continue reasoning until it reaches a final answer.", metadata={"source": "agents.txt"}),
    Document(page_content="Prompt templates in LangChain allow you to build reusable prompts with placeholders. ChatPromptTemplate supports system, human, and AI message roles. MessagesPlaceholder injects full chat history into a prompt.", metadata={"source": "prompts.txt"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(raw_docs)

print(f"\n[Document Loader + Text Splitter]")
print(f"Original docs: {len(raw_docs)}")
print(f"After splitting: {len(chunks)} chunks")
print(f"First chunk: {chunks[0].page_content}")
print(f"First chunk metadata: {chunks[0].metadata}")


# ════════════════════════════════════════════════════════════
# SECTION 7 — VECTOR STORE + RETRIEVER + RAG
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 7 — VECTOR STORE + RETRIEVER + RAG")
print("="*60)

# Create Chroma vector store from chunks
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="revision_collection"
)

# Basic similarity search
query = "What is RAG?"
sim_results = vector_store.similarity_search(query, k=2)
print(f"\n[Similarity Search for '{query}']")
for i, doc in enumerate(sim_results):
    print(f"Result {i+1}: {doc.page_content}")

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# RAG Chain
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If not found in context, say "I don't know from the provided documents."

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

rag_result = rag_chain.invoke("How do agents work in LangChain?")
print(f"\n[RAG Chain Output]")
print(rag_result)


# ════════════════════════════════════════════════════════════
# SECTION 8 — TOOLS + TOOL CALLING
# ════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 8 — TOOLS + TOOL CALLING")
print("="*60)

# Define tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result"""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Get the current weather description for a given city name"""
    weather_data = {
        "mumbai": "Hot and humid, 34°C",
        "delhi": "Sunny, 38°C",
        "bangalore": "Pleasant, 26°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

print(f"\n[Tool Properties]")
print(f"add.name: {add.name}")
print(f"add.description: {add.description}")
print(f"add.args: {add.args}")

# Bind tools to model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)
llm_with_tools = llm.bind_tools([add, multiply, get_weather])

# Tool calling loop
messages = [HumanMessage("What is 15 multiplied by 7?")]
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)

print(f"\n[Tool Calls Requested by Model]")
print(ai_response.tool_calls)

# Run the tools
tool_map = {"add": add, "multiply": multiply, "get_weather": get_weather}

for tool_call in ai_response.tool_calls:
    tool_fn = tool_map[tool_call["name"]]
    tool_result = tool_fn.invoke(tool_call)
    messages.append(tool_result)
    print(f"\n[Tool '{tool_call['name']}' Result]: {tool_result.content}")

# Final answer
final_answer = llm_with_tools.invoke(messages).content
print(f"\n[Final Answer from Model]")
print(final_answer)

print("\n" + "="*60)
print("REVISION COMPLETE — All concepts covered!")
print("="*60)