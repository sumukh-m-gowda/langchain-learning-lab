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