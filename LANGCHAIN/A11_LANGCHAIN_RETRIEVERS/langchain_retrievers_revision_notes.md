# LangChain Retrievers – Complete Revision Notes

---

## What are Retrievers?

In the last lecture you learned how to store documents in Chroma and search them using `.similarity_search()`. A **Retriever** is a wrapper around that search — it gives you a clean, standard `.invoke(query)` interface that plugs directly into a LangChain chain.

The difference is simple. With a vector store you call `.similarity_search(query, k=2)` directly. With a retriever you call `.invoke(query)` — the same method used everywhere else in LangChain, which means you can pipe it into a chain with `|`.

This is **step 5 of the RAG pipeline**: Load → Split → Embed → Store → **Retrieve** → Answer.

There are four types of retrievers covered in this lecture:

1. WikipediaRetriever — fetches documents directly from Wikipedia
2. Vector Store Retriever — wraps a Chroma or FAISS vector store
3. MMR Retriever — adds diversity to your search results
4. MultiQueryRetriever — generates multiple versions of your query for better recall
5. ContextualCompressionRetriever — strips irrelevant content from retrieved chunks

---

## Setup

```python
!pip install langchain langchain-google-genai langchain-chroma langchain-community faiss-cpu wikipedia python-dotenv
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
```

Your `.env` file needs `GEMINI_API_KEY=your_key_here`.

---

## 1. WikipediaRetriever

The simplest retriever — no vector store needed. It directly queries Wikipedia and returns the most relevant articles as Document objects. Useful when you want real-world factual knowledge without setting up an embedding pipeline first.

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
```

`top_k_results=2` means return the top 2 Wikipedia articles. `lang="en"` sets the language. The retriever handles everything — searching Wikipedia, fetching the page content, and returning it as Document objects with metadata. You just call `.invoke(query)` and get results back.

---

## 2. Vector Store Retriever

This is the standard way to turn your Chroma or FAISS vector store into a retriever. Instead of calling `.similarity_search()` directly on the vector store, you call `.as_retriever()` on it and get back a retriever object.

```python
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="Gemini provides powerful embedding models."),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
```

`Chroma.from_documents()` is a shortcut that creates the vector store and adds documents in one step, instead of calling `Chroma(...)` then `.add_documents()` separately as you did in the last lecture.

`.as_retriever(search_kwargs={"k": 2})` wraps the vector store. `search_kwargs` is how you pass parameters like `k` into the underlying similarity search.

After this, calling `retriever.invoke(query)` does exactly the same thing as `vectorstore.similarity_search(query, k=2)` — the difference is the retriever version plugs cleanly into chains.

---

## 3. MMR Retriever (Maximum Marginal Relevance)

The problem with standard similarity search is that if you ask for `k=3` results, you might get three documents that all say almost the same thing — they're all similar to your query but also similar to each other. MMR solves this by balancing **relevance** and **diversity** in the results.

MMR picks the first result normally (most similar to query). For each subsequent result, it picks the document that is most relevant to the query but least similar to what was already picked. The result set is relevant but not repetitive.

```python
from langchain_community.vectorstores import FAISS

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
```

`search_type="mmr"` switches from standard similarity to MMR mode. `lambda_mult` controls the relevance-diversity balance — `0.5` is equal balance, `1.0` is pure similarity (same as standard search), `0.0` is pure diversity with no regard for relevance.

Notice **FAISS** is used here instead of Chroma. FAISS is another vector store — it runs entirely in memory (no disk persistence) and is very fast. The `.as_retriever()` interface works the same way for both Chroma and FAISS.

Without MMR, a query about "langchain" might return the first two LangChain documents since they're the most similar. With MMR, the third result would be something different (like the Chroma or embeddings document) to give you a broader picture.

---

## 4. MultiQueryRetriever

Standard similarity search has a limitation — it only searches for documents similar to the exact phrasing of your query. If you phrase something differently or your query has multiple aspects, you might miss relevant documents.

MultiQueryRetriever solves this by using an LLM to automatically generate several different versions of your original query, running each one as a separate search, then combining and deduplicating all the results.

```python
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=model
)

query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Similarity Result {i+1} ---")
    print(doc.page_content)

print("*" * 150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- MultiQuery Result {i+1} ---")
    print(doc.page_content)
```

The dataset is intentionally tricky — documents H1 to H5 are about health, but documents I1 to I5 also contain the words "energy" and "balance" in completely different contexts (solar energy, Python balancing readability, etc.). A simple similarity search on "improve energy levels and maintain balance" might pick up those irrelevant documents because the words match.

MultiQueryRetriever generates variations like "how to boost energy naturally", "ways to maintain physical balance", "tips for staying energised" — and by searching across multiple angles, it finds the genuinely health-related documents more reliably while the irrelevant matches average out and get dropped.

`MultiQueryRetriever.from_llm()` takes the base retriever and the LLM. The LLM's only job here is to rephrase the query — it does not generate the final answer.

---

## 5. ContextualCompressionRetriever

Standard retrieval fetches whole chunks. But often a chunk contains a mix of relevant and irrelevant content — for example, a paragraph about the Grand Canyon that also happens to mention photosynthesis. If you ask about photosynthesis, you get the whole paragraph even though most of it is irrelevant noise.

ContextualCompressionRetriever fixes this by adding a second step: after retrieving the chunks normally, it passes each chunk to an LLM and asks it to extract only the parts that are actually relevant to the query. Everything else is stripped out.

```python
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

docs = [
    Document(page_content=(
        "The Grand Canyon is one of the most visited natural wonders in the world. "
        "Photosynthesis is the process by which green plants convert sunlight into energy. "
        "Millions of tourists travel to see it every year. The rocks date back millions of years."
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        "In medieval Europe, castles were built primarily for defense. "
        "The chlorophyll in plant cells captures sunlight during photosynthesis. "
        "Knights wore armor made of metal. Siege weapons were often used to breach castle walls."
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        "Basketball was invented by Dr. James Naismith in the late 19th century. "
        "It was originally played with a soccer ball and peach baskets. NBA is now a global league."
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        "The history of cinema began in the late 1800s. Silent films were the earliest form. "
        "Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells. "
        "Modern filmmaking involves complex CGI and sound design."
    ), metadata={"source": "Doc4"})
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.from_documents(docs, embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
```

The documents here are intentionally messy — each one is about a completely different topic (Grand Canyon, medieval castles, basketball, cinema) but has a sentence about photosynthesis buried inside it. The basketball document has nothing about photosynthesis at all.

Without compression: you get back all four documents with all the irrelevant content included.

With compression: `LLMChainExtractor` reads each retrieved chunk and extracts only the sentence(s) about photosynthesis. The basketball document returns nothing (since there is nothing relevant in it to extract). The other three return only their photosynthesis sentences, stripped of all the unrelated content about tourism, knights, and cinema.

`base_retriever` does the initial fetch. `base_compressor` does the filtering. `ContextualCompressionRetriever` wraps both together into one `.invoke()` call.

---

## Comparison of All Five Retrievers

| Retriever | What makes it different | Best for |
|---|---|---|
| WikipediaRetriever | No setup needed, fetches from Wikipedia directly | Quick factual Q&A, prototyping |
| Vector Store Retriever | Standard similarity search on your own documents | Most RAG applications |
| MMR Retriever | Balances relevance and diversity in results | When top results are too similar to each other |
| MultiQueryRetriever | Generates multiple query variations for better recall | Complex queries, ambiguous phrasing |
| ContextualCompressionRetriever | Strips irrelevant content from retrieved chunks | Noisy documents with mixed topics |

---

## Key Concepts to Remember

A retriever is just a wrapper around a vector store that gives it the standard `.invoke()` interface so it plugs into chains with `|`.

`.as_retriever()` converts any vector store (Chroma or FAISS) into a retriever. `search_kwargs={"k": 2}` is how you pass the number of results.

`Chroma.from_documents()` and `FAISS.from_documents()` are one-step shortcuts that create the vector store and embed the documents together.

FAISS is an in-memory vector store (fast, no disk persistence). Chroma persists to disk with `persist_directory`. Use FAISS for quick experiments, Chroma for production apps where data must survive restarts.

MMR's `lambda_mult` controls the relevance-diversity trade-off — 0.5 is the balanced default, 1.0 is pure relevance, 0.0 is pure diversity.

MultiQueryRetriever uses the LLM only to rephrase the query into variations — the LLM does not answer the question here.

ContextualCompressionRetriever is the most token-expensive because it sends every retrieved chunk back to the LLM for filtering. Use it when document quality is more important than speed or cost.

After retrievers, the next and final step of RAG is feeding the retrieved documents into a prompt and getting the model to answer — that is the full RAG chain.
