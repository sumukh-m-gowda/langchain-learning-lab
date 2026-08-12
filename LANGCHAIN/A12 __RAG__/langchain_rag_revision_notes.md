# LangChain RAG (Retrieval-Augmented Generation) – Complete Revision Notes

---

## What is RAG?

Every LLM has a knowledge cutoff — it was trained on data up to a certain date and knows nothing about your private documents, your company's internal data, or anything that happened after training. If you ask GPT-4 about your own PDF or a document you uploaded, it simply does not know.

RAG solves this by giving the model external knowledge at the time of answering. Instead of relying on what the model memorised during training, you first retrieve the relevant pieces of your own documents and inject them into the prompt. The model then answers based on that retrieved context, not from memory.

In short: RAG = Retrieve relevant documents first → Augment the prompt with them → Generate the answer.

---

## The Full RAG Pipeline

Every concept you have studied so far was building towards this pipeline:

```
1. LOAD       → Document Loaders     (TextLoader, PyPDFLoader, WebBaseLoader)
2. SPLIT      → Text Splitters       (RecursiveCharacterTextSplitter)
3. EMBED      → Embedding Models     (GoogleGenerativeAIEmbeddings)
4. STORE      → Vector Store         (Chroma / FAISS)
5. RETRIEVE   → Retrievers           (as_retriever, MMR, MultiQuery)
6. ANSWER     → Prompt + Model + Parser  (ChatPromptTemplate | model | StrOutputParser)
```

Steps 1 to 5 you have already studied individually. RAG just connects all of them into one working chain.

---

## Setup

```python
pip install langchain langchain-google-genai langchain-chroma langchain-community pypdf python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
# .env file needs: GEMINI_API_KEY=your_key_here
```

---

## Step 1: Load

Load your source document — this is the knowledge base the model will answer from.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('your_document.pdf')
docs = loader.load()

print(f"Loaded {len(docs)} pages")
```

---

## Step 2: Split

The loaded document is too large to fit into a prompt directly. Split it into smaller chunks.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks")
```

`chunk_overlap=200` is important here — a higher overlap than in the text splitters lecture because in RAG, losing context at a chunk boundary means giving the model an incomplete answer.

---

## Step 3: Embed and Store

Convert all chunks into vectors and store them in Chroma. This only needs to be done once — because of `persist_directory`, the database stays on disk and you can reload it without re-embedding.

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="rag_db",
    collection_name="my_docs"
)

print(f"Stored {vector_store._collection.count()} chunks in vector store")
```

---

## Step 4: Create the Retriever

Wrap the vector store into a retriever so it can plug into the chain.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

`k=3` means fetch the 3 most relevant chunks for any given question. You can increase this for more context but it uses more tokens.

---

## Step 5: Build the RAG Chain

This is where everything comes together. The chain takes a user question, retrieves relevant chunks, injects them into the prompt as context, and generates an answer.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not found in the context, say "I don't know based on the provided documents."

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
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("What is this document about?")
print(result)
```

---

## How the RAG Chain Flows — Step by Step

This is the most important part to understand.

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)
```

When you call `rag_chain.invoke("What is this document about?")`, here is exactly what happens:

The input string `"What is this document about?"` enters the chain. The first step is a dictionary with two keys. For the `"context"` key, the question is sent to the retriever which fetches the top 3 most relevant chunks from the vector store, then `format_docs` joins them into one big string with double newlines between chunks. For the `"question"` key, `RunnablePassthrough()` passes the original question through unchanged (you saw this in the Runnables lecture — it just passes input forward as-is).

Now the prompt has both `{context}` (the retrieved chunks) and `{question}` (the original question) filled in. The model receives this combined prompt and generates an answer based only on what was retrieved. `StrOutputParser()` extracts the plain text from the `AIMessage` response.

The model never looks into its training data to answer — it only uses what was retrieved from your documents.

---

## The format_docs Function

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

The retriever returns a list of Document objects. The prompt template expects a plain string for `{context}`. This function converts the list of Document objects into one continuous string by joining all `page_content` values with a blank line between each chunk. Without this, the prompt would receive a Python list object instead of readable text.

---

## Loading an Existing Vector Store (Without Re-embedding)

If you have already embedded and stored your documents with `persist_directory`, you do not need to run the whole pipeline again. Just load it back:

```python
vector_store = Chroma(
    persist_directory="rag_db",
    embedding_function=embedding_model,
    collection_name="my_docs"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

This is why `persist_directory` matters — embedding thousands of documents is expensive and slow. You do it once, save to disk, and reload instantly every time after that.

---

## Complete RAG Script (All Steps Together)

```python
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── 1. LOAD ──────────────────────────────────────────
loader = PyPDFLoader('your_document.pdf')
docs = loader.load()

# ── 2. SPLIT ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# ── 3. EMBED + STORE ──────────────────────────────────
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="rag_db",
    collection_name="my_docs"
)

# ── 4. RETRIEVE ───────────────────────────────────────
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ── 5. PROMPT + MODEL ─────────────────────────────────
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not found in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 6. CHAIN ──────────────────────────────────────────
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# ── INVOKE ────────────────────────────────────────────
question = "What is this document about?"
answer = rag_chain.invoke(question)
print(answer)
```

---

## Why "Answer only from context" in the Prompt Matters

Notice the prompt says: "Answer the question using only the context provided below. If the answer is not found in the context, say I don't know."

Without this instruction, the model will mix retrieved context with its own training knowledge and hallucinate confident-sounding answers about things not in your document. This instruction forces the model to stay grounded in only what was retrieved — which is the whole point of RAG.

---

## Where Every Previous Concept Connects to RAG

| Concept Learned | Where it appears in RAG |
|---|---|
| ChatGoogleGenerativeAI (Models) | The final answer-generating model |
| ChatPromptTemplate (Prompts) | The prompt with `{context}` and `{question}` |
| StrOutputParser (Output Parsers) | Extracts plain text from the model's response |
| RunnablePassthrough (Runnables) | Passes the question through to the prompt unchanged |
| `\|` pipe / RunnableSequence (Chains) | Connects retriever → format_docs → prompt → model → parser |
| PyPDFLoader (Document Loaders) | Loads the source document |
| RecursiveCharacterTextSplitter (Text Splitters) | Breaks document into chunks |
| GoogleGenerativeAIEmbeddings (Models) | Converts chunks and queries into vectors |
| Chroma (Vector Store) | Stores and searches the embeddings |
| as_retriever() (Retrievers) | Fetches the relevant chunks for each question |

---

## Key Concepts to Remember

RAG stands for Retrieval-Augmented Generation. It lets the model answer questions from your own documents instead of only from its training data.

The pipeline order is always: Load → Split → Embed → Store → Retrieve → Answer. You cannot skip or reorder these steps.

`format_docs` is a simple but essential helper — it converts the list of retrieved Document objects into a plain string the prompt can use.

`RunnablePassthrough()` in the RAG chain keeps the original question flowing through to the prompt while the retriever is simultaneously fetching context. You studied this exact pattern in the Runnables lecture.

The `persist_directory` in Chroma means you embed once and reuse forever. Never re-embed the same documents unnecessarily.

`chunk_overlap` should be higher in RAG (150-200) than in basic text splitting because losing context at boundaries directly impacts the quality of the model's answers.

Always instruct the model to answer only from the provided context. Without this, the model will mix retrieved content with hallucinated training knowledge.

`k` in `search_kwargs={"k": 3}` controls how many chunks are retrieved. More chunks = more context but more tokens used. 3 to 5 is a good starting range.
