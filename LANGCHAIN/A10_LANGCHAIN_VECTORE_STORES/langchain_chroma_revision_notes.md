# LangChain Chroma Vector Store – Complete Revision Notes

---

## Why Vector Stores?

You already know that embeddings convert text into vectors, and cosine similarity can find the most similar document for a query. But until now, you were computing this manually with `sklearn`. In a real application, you might have thousands or millions of documents, and you cannot loop through embeddings manually every time.

A vector store is a database built specifically to store embeddings and search through them efficiently. **Chroma** is one of the most popular open-source vector stores, and it integrates directly with LangChain.

This is **step 4 of the RAG pipeline**: Load → Split → Embed → **Store** → Retrieve → Answer.

---

## Setup

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

```bash
pip install langchain chromadb openai tiktoken pypdf langchain_openai langchain-community
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
```

---

## Step 1: Create Documents

Just like before, data is represented as `Document` objects with `page_content` and `metadata`. Here, five IPL player documents are created, each tagged with their team in the metadata.

```python
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]
```

The `metadata` here is important — it is what lets you filter results later by team, without changing the text content itself.

---

## Step 2: Create the Chroma Vector Store

```python
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)
```

`embedding_function=OpenAIEmbeddings()` tells Chroma which embedding model to use to convert text into vectors whenever documents are added or a query is searched.

`persist_directory='my_chroma_db'` is where Chroma saves the database to disk. This means the data survives even after you restart your script — you don't need to re-embed everything every time.

`collection_name='sample'` is the name of this particular collection inside the database. You can have multiple collections in the same persisted directory, like separate tables in a database.

---

## Step 3: Add Documents

```python
vector_store.add_documents(docs)
```

This single line does a lot of work behind the scenes: it takes each document's `page_content`, runs it through `OpenAIEmbeddings()` to generate a vector, and stores the vector along with the original text and metadata in the Chroma database. Each document automatically gets a unique ID.

---

## Step 4: View All Documents

```python
vector_store.get(include=['embeddings', 'documents', 'metadatas'])
```

`.get()` retrieves everything stored in the vector store. The `include` parameter lets you choose what to see in the output — the raw embedding vectors, the original text content, and the metadata. Useful for debugging and confirming your data was stored correctly.

---

## Step 5: Similarity Search

This is the core operation — given a query, find the most relevant documents.

```python
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)
```

`query` is the natural language question. `k=2` means return the top 2 most similar documents. Internally, Chroma embeds the query, compares it against every stored vector using similarity, and returns the closest matches — this replaces the manual cosine similarity code you wrote earlier with `sklearn`.

For this query, it would correctly return the Jasprit Bumrah document as the top result, since he's the bowler among the five players.

---

## Step 6: Similarity Search with Score

```python
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
```

Same as above, but also returns the similarity score alongside each document. This is useful when you want to see *how confident* the match is, or when you want to filter out results below a certain score threshold.

---

## Step 7: Metadata Filtering

```python
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)
```

Instead of (or in addition to) searching by meaning, you can filter results by metadata fields directly. Here, `filter={"team": "Chennai Super Kings"}` returns only documents belonging to Chennai Super Kings — both MS Dhoni and Ravindra Jadeja, regardless of similarity to the (empty) query.

This combination of semantic search and metadata filtering is very powerful in real applications — for example, "find documents about refunds, but only from the billing department."

---

## Step 8: Update a Document

```python
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)
```

`update_document()` replaces an existing document with new content, using its unique `document_id` (which you get from `.get()`). Behind the scenes, Chroma re-embeds the new content and replaces the old vector. This is important because if you only changed the text without updating the embedding, similarity search would still be matching against the old, outdated meaning.

---

## Step 9: Delete a Document

```python
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])
```

Removes a document (and its embedding) from the vector store permanently, using its ID. After this, the document will no longer appear in `.get()` or `similarity_search()` results.

---

## Summary of All Chroma Operations

| Operation | Method | What it does |
|---|---|---|
| Create store | `Chroma(embedding_function=..., persist_directory=..., collection_name=...)` | Initialises (or loads) a vector database |
| Add | `.add_documents(docs)` | Embeds and stores a list of Documents |
| View all | `.get(include=[...])` | Retrieves everything stored, with optional fields |
| Search | `.similarity_search(query, k)` | Returns top-k most similar documents |
| Search + score | `.similarity_search_with_score(query, k)` | Same as above, with similarity scores |
| Filter | `.similarity_search_with_score(query, filter={...})` | Filters results by metadata |
| Update | `.update_document(document_id, document)` | Replaces a document and re-embeds it |
| Delete | `.delete(ids=[...])` | Permanently removes a document |

---

## Key Concepts to Remember

A vector store like Chroma replaces the manual `sklearn` cosine similarity code from the Models lecture — it does the embedding, storage, and similarity search for you, and it scales to much larger datasets.

`persist_directory` is what makes the database permanent on disk instead of disappearing when your script ends.

Each document gets stored with three things: its vector (embedding), its original text (`page_content`), and its `metadata`.

Metadata filtering and similarity search can be combined — this is extremely useful for real applications where you want both semantic relevance and structured filtering (like team, date, category, department).

Updating a document re-embeds it — this matters because the vector must always reflect the current text, not the old one.

Document IDs are required for both updating and deleting — get them first using `.get()`.

This is the vector store step of RAG. Once documents are stored here, the next step is building a **Retriever** that automatically fetches relevant chunks for a user's question and feeds them into a prompt before sending it to the model.
