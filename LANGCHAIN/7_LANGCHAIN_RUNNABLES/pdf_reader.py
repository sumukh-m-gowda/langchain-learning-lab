from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate   
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv
import os

load_dotenv()

# Load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    dimensions=32,
    google_api_key=os.getenv("GEMINI_API_KEY_embedding")
)

vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever
retriever = vectorstore.as_retriever()

# Initialize Gemini
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create RetrievalQA Chain (just to join all the retrivals from the previous output)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Ask a question
query = "What are the key takeaways from the document?"

answer = qa_chain.invoke(query)

print("Answer:", answer)