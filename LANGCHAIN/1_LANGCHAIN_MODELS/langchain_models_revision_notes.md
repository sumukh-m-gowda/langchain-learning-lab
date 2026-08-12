# LangChain Models – Complete Revision Notes

---

## What is LangChain?

LangChain is a Python framework that makes it easy to build applications powered by large language models (LLMs). It provides a unified interface so you can swap between different model providers (OpenAI, Anthropic, Google, HuggingFace, etc.) without rewriting your core logic.

To check your installed version:

```python
import langchain
print(langchain.__version__)
```

---

## Setup: Environment Variables

All API keys (OpenAI, Anthropic, Google, HuggingFace) are stored in a `.env` file and loaded using `python-dotenv`. This keeps secrets out of your source code.

```python
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` reads the `.env` file in your project root and injects the keys as environment variables, which LangChain picks up automatically.

---

## The Three Types of Models in LangChain

LangChain organises models into three categories:

1. **LLMs** – older-style completion models (text in → text out)
2. **Chat Models** – modern conversational models (messages in → message out)
3. **Embedding Models** – convert text into numeric vectors for similarity search

---

## 1. LLMs (Completion Models)

These are the older generation of models that take a plain string prompt and return a plain string. OpenAI's `gpt-3.5-turbo-instruct` is an example.

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of India")

print(result)
```

The `.invoke()` method sends the prompt and returns the raw text response as a string.

---

## 2. Chat Models

Chat models are the modern standard. They accept a list of messages (with roles like `user`, `assistant`, `system`) and return a message object. You access the text via `.content`.

### Key Parameters

- `model` – which model version to use
- `temperature` – controls randomness (0 = deterministic, 2 = very random)
- `max_completion_tokens` – limits the length of the response

### 2.1 OpenAI Chat Model

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
```

### 2.2 Anthropic Chat Model (Claude)

```python
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke('What is the capital of India')

print(result.content)
```

### 2.3 Google Gemini Chat Model

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

result = model.invoke('What is the capital of India')

print(result.content)
```

### 2.4 HuggingFace Chat Model via API (Cloud Inference)

This uses HuggingFace's hosted inference API. You don't need the model locally – it runs on HuggingFace's servers. `HuggingFaceEndpoint` wraps the API call, and `ChatHuggingFace` wraps that to give it the chat interface.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

### 2.5 HuggingFace Chat Model Locally (Downloaded to Disk)

This downloads the model weights to your machine and runs inference locally using a pipeline. `HF_HOME` controls where the weights are cached.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
```

The difference between 2.4 and 2.5:
- 2.4 (API) sends your input to HuggingFace's servers – needs internet, no local GPU required
- 2.5 (Local) downloads the model weights to your disk and runs everything offline – needs enough RAM/GPU

---

## 3. Embedding Models

Embedding models convert text into a list of floating-point numbers called a **vector**. Semantically similar texts produce vectors that are mathematically close to each other. This is the foundation of semantic search and RAG (Retrieval-Augmented Generation).

### 3.1 Embed a Single Query (OpenAI)

`embed_query()` takes a single string and returns one vector.

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))
```

`dimensions=32` reduces the vector to 32 numbers (instead of the default 3072). Smaller vectors are faster to store and compare but capture less detail.

### 3.2 Embed Multiple Documents (OpenAI)

`embed_documents()` takes a list of strings and returns a list of vectors (one per document).

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))
```

### 3.3 HuggingFace Embeddings (Local)

Uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight open-source model that runs locally. No API key needed.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))
```

### 3.4 Document Similarity using Cosine Similarity

This is the most important embedding use case. Given a user query, find the most semantically similar document from a list.

Steps:
1. Embed all documents into vectors
2. Embed the query into a vector
3. Compute cosine similarity between the query vector and each document vector
4. Return the document with the highest score

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)
```

How cosine similarity works: it measures the angle between two vectors. A score of 1.0 means identical direction (very similar meaning), 0.0 means no relation, and -1.0 means opposite. The query "tell me about bumrah" correctly returns the Bumrah document with the highest score even though there is no exact word match – the model understands semantic meaning.

---

## Required Packages (requirements.txt)

```
# LangChain Core
langchain
langchain-core

# OpenAI Integration
langchain-openai
openai

# Anthropic Integration
langchain-anthropic

# Google Gemini Integration
langchain-google-genai
google-generativeai

# HuggingFace Integration
langchain-huggingface
transformers
huggingface-hub

# Environment Variable Management
python-dotenv

# Machine Learning Utilities
numpy
scikit-learn
```

Install with: `pip install -r requirements.txt`

---

## Quick Comparison Table

| Type | Class | Input | Output | Use Case |
|---|---|---|---|---|
| LLM | `OpenAI` | plain string | plain string | simple completion |
| Chat Model | `ChatOpenAI`, `ChatAnthropic`, etc. | string or messages | message object (`.content`) | conversations, Q&A |
| Embedding | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` | string or list | vector / list of vectors | search, similarity |

---

## Key Concepts to Remember

- Always call `load_dotenv()` before initialising any model that needs an API key.
- Chat models return a message object – you must use `.content` to get the text string.
- LLMs (old style) return a plain string directly from `.invoke()`.
- `embed_query()` is for a single string; `embed_documents()` is for a list.
- `dimensions` in OpenAI embeddings lets you control the vector size for a trade-off between speed and accuracy.
- Cosine similarity is the standard way to find the most relevant document for a query.
- HuggingFace models can run via API (no local setup) or locally (download weights with `HuggingFacePipeline`).
- `HF_HOME` sets where HuggingFace caches downloaded model weights locally.
