# LangChain Revision Project — Detailed Explanation Notes

This file explains every single section of `langchain_revision_project.py` line by line. Read this alongside the code.

---

## Setup

```python
from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

`load_dotenv()` reads your `.env` file and injects all keys as environment variables. `os.getenv("GEMINI_API_KEY")` fetches the Gemini API key from that environment. Every model in this project uses this key — you only need to set it once here.

Your `.env` file should look like:
```
GEMINI_API_KEY=your_key_here
```

---

## Section 1 — Models

```python
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)
result = chat_model.invoke("What is LangChain in one sentence?")
print(result.content)
```

`ChatGoogleGenerativeAI` is your chat model — it takes messages and returns a message object. `temperature=0.3` keeps responses focused and consistent rather than creative. `.invoke()` sends the prompt and returns an `AIMessage` object. You must call `.content` on it to get the plain text string — the raw object is not a string.

```python
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
vector = embedding_model.embed_query("LangChain is a framework for LLMs")
print(f"Vector length: {len(vector)}, First 3 values: {vector[:3]}")
```

`GoogleGenerativeAIEmbeddings` converts text into a list of floating-point numbers called a vector. `embed_query()` takes a single string and returns one vector. The length of the vector tells you how many dimensions the model uses. This same embedding model is reused in Section 7 for the vector store — it must be the same model used for both storing and querying, otherwise similarity scores won't be comparable.

---

## Section 2 — Prompts

```python
prompt = ChatPromptTemplate([
    ("system", "You are an expert in {domain}. Answer concisely."),
    ("human", "{question}")
])
chain = prompt | chat_model | StrOutputParser()
result = chain.invoke({"domain": "cricket", "question": "Who has scored the most runs in IPL?"})
```

`ChatPromptTemplate` takes a list of `(role, content)` tuples. Both the system message and the human message have `{placeholders}` that get filled at `.invoke()` time. The `|` pipe creates a chain: template fills placeholders → model generates a response → `StrOutputParser` extracts `.content` from the `AIMessage`. Without `StrOutputParser` at the end, `result` would be an `AIMessage` object, not a string.

```python
history_prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chat_history = [
    HumanMessage(content="My name is Rahul"),
    AIMessage(content="Nice to meet you Rahul!")
]

result = history_chain.invoke({
    "chat_history": chat_history,
    "question": "Do you remember my name?"
})
```

`MessagesPlaceholder` reserves a slot in the template for an entire list of messages. When you pass `chat_history` as a list of `HumanMessage` and `AIMessage` objects, they get injected at that slot. The model receives the full conversation history and can answer "Do you remember my name?" correctly because it sees "My name is Rahul" from earlier in the history. This is the foundation of any stateful chatbot.

---

## Section 3 — Structured Output

```python
class MovieReview(BaseModel):
    title: str = Field(description="Name of the movie")
    sentiment: Literal["positive", "negative", "neutral"] = Field(...)
    rating: int = Field(gt=0, lt=11, description="Rating out of 10")
    pros: list[str] = Field(description="List of positive aspects")
    cons: Optional[list[str]] = Field(default=None, description="List of negative aspects if any")
    summary: str = Field(description="One line summary of the review")
```

This is a Pydantic `BaseModel` that defines the exact shape of output you want. Each field has a `description` — this is what the model reads to understand what to fill in. `Literal["positive", "negative", "neutral"]` restricts `sentiment` to exactly those three values. `Field(gt=0, lt=11)` means `rating` must be between 1 and 10 — Pydantic validates this at runtime and will raise an error if violated. `Optional[list[str]]` means `cons` can be a list or `None` — useful when the review might not mention any negatives.

```python
structured_model = chat_model.with_structured_output(MovieReview)
result = structured_model.invoke(review_text)
print(result.rating)  # dot notation because it's a Pydantic object
```

`.with_structured_output(MovieReview)` wraps the model so its output is forced to match the schema. The result is a `MovieReview` Pydantic object — you access fields with dot notation like `result.sentiment`, `result.pros`. This is different from `PydanticOutputParser` which uses format instructions injected into the prompt — `.with_structured_output()` uses the model's native function-calling capability directly.

---

## Section 4 — Output Parsers

```python
parser = PydanticOutputParser(pydantic_object=Person)

parser_prompt = PromptTemplate(
    template="Generate a fictional Indian person's details.\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

parser_chain = parser_prompt | chat_model | parser
result = parser_chain.invoke({})
```

`PydanticOutputParser` works differently from `.with_structured_output()`. It injects format instructions into the prompt text itself — `parser.get_format_instructions()` returns a string that tells the model exactly how to format the JSON output. This string goes into `partial_variables` because it never changes between calls (unlike `input_variables` which are user-supplied at `.invoke()` time). The parser at the end of the chain reads the model's JSON text output and converts it into a `Person` Pydantic object. `input_variables=[]` because the only variable in this prompt is `format_instruction` which is already filled via `partial_variables`.

---

## Section 5 — Chains + Runnables

```python
sequential_chain = template1 | chat_model | StrOutputParser() | template2 | chat_model | StrOutputParser()
result = sequential_chain.invoke({"topic": "monsoon"})
```

This is a two-step sequential chain. Step 1: `template1` fills `{topic}` and the model writes a poem — `StrOutputParser` extracts it as a plain string. Step 2: that plain string flows directly into `template2` as the value for `{poem}`, the model translates it, and the final `StrOutputParser` extracts the translated text. `StrOutputParser` is the glue here — without it, the `AIMessage` from step 1 would flow into `template2` which expects a plain string, causing an error.

```python
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})
full_chain = joke_chain | parallel_chain
result = full_chain.invoke({"topic": "programmers"})
```

`joke_chain` generates a joke string. That string then enters `RunnableParallel` which sends it to two places at the same time. `RunnablePassthrough()` takes the joke string and passes it forward unchanged — it preserves the original joke in the result dict under the `"joke"` key. `RunnableLambda(word_count)` takes the same joke string, runs the `word_count` function on it, and puts the integer count under `"word_count"`. The final `result` is `{"joke": "...", "word_count": 17}`. Without `RunnablePassthrough`, the joke would only flow into `word_count` and you'd lose the joke text entirely.

---

## Section 6 — Document Loaders + Text Splitters

```python
raw_docs = [
    Document(page_content="LangChain is a framework...", metadata={"source": "intro.txt"}),
    ...
]
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(raw_docs)
```

In a real project you'd use `PyPDFLoader` or `WebBaseLoader` to load actual files. Here raw `Document` objects are created directly to avoid needing a specific file. Each Document has `page_content` (the text) and `metadata` (info about the source). `RecursiveCharacterTextSplitter` splits each document into chunks of at most 200 characters, trying to break at paragraph boundaries first, then line breaks, then spaces. `chunk_overlap=30` means 30 characters are repeated between consecutive chunks so context is not lost at boundaries. `split_documents()` is used instead of `split_text()` because the input is Document objects — this method preserves the metadata from the originals.

---

## Section 7 — Vector Store + Retriever + RAG

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="revision_collection"
)
```

`Chroma.from_documents()` is a one-step shortcut — it embeds every chunk using `embedding_model` and stores the vectors in a Chroma collection. This replaces the two-step `Chroma(...)` then `.add_documents()` pattern. No `persist_directory` is set here so it lives in memory — in a real project you'd add `persist_directory="rag_db"` to save to disk and avoid re-embedding on every restart.

```python
sim_results = vector_store.similarity_search(query, k=2)
```

This searches the vector store directly. It embeds the query, compares against all stored vectors, and returns the top 2 most similar Document objects. This replaces the manual `sklearn` cosine similarity code from the Models lecture.

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

`.as_retriever()` wraps the vector store into a retriever — the difference is the retriever uses `.invoke()` instead of `.similarity_search()`, which means it can plug directly into a chain with `|`.

```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | chat_model
    | StrOutputParser()
)
result = rag_chain.invoke("How do agents work in LangChain?")
```

This is the full RAG chain. When `.invoke("How do agents work in LangChain?")` is called, the input string enters the first step which is a dictionary. For the `"context"` key: the question goes to the retriever which fetches the 2 most relevant chunks, then `format_docs` joins those Document objects into one plain string. For the `"question"` key: `RunnablePassthrough()` passes the original question through unchanged. Now the prompt has both `{context}` and `{question}` filled. The model answers based only on the retrieved context, not from training memory. `StrOutputParser()` extracts the final plain text answer.

The prompt says "If not found in context, say I don't know from the provided documents" — this prevents the model from mixing retrieved content with hallucinated training knowledge, which is the whole point of RAG.

---

## Section 8 — Tools + Tool Calling

```python
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result"""
    return a * b
```

The `@tool` decorator turns a plain Python function into a LangChain tool. It reads the function name as `tool.name`, the docstring as `tool.description`, and the type hints to build `tool.args` automatically. The docstring is critical — the model reads it to decide when to use this tool. A bad or missing docstring means the model won't know when to call it.

```python
llm_with_tools = llm.bind_tools([add, multiply, get_weather])
```

`bind_tools()` registers the tools with the model. The model now knows what tools exist and what each one does. Without this, the model cannot call any tool even if you define them.

```python
messages = [HumanMessage("What is 15 multiplied by 7?")]
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)
print(ai_response.tool_calls)
```

The model does not answer "105" directly. Instead it returns an `AIMessage` with `tool_calls` populated — a list like `[{"name": "multiply", "args": {"a": 15, "b": 7}, "id": "..."}]`. The model is saying "please call multiply with these arguments." It has not run the tool itself — LLMs cannot execute code. You append this `AIMessage` to the messages list so the model has context about what it already decided.

```python
tool_map = {"add": add, "multiply": multiply, "get_weather": get_weather}

for tool_call in ai_response.tool_calls:
    tool_fn = tool_map[tool_call["name"]]
    tool_result = tool_fn.invoke(tool_call)
    messages.append(tool_result)
```

You loop through each tool call, find the matching function in `tool_map`, run it with the args the model provided, and append the result as a `ToolMessage` to the messages list. The `tool_map` dictionary is just a clean way to look up the right function by name. Each `ToolMessage` contains the tool's output and is linked to the original tool call by ID.

```python
final_answer = llm_with_tools.invoke(messages).content
```

Now the messages list contains: `HumanMessage` → `AIMessage(tool_calls)` → `ToolMessage(result)`. The model sees the full picture — what was asked, what tool it decided to call, and what the tool returned. It now generates a final human-readable answer like "15 multiplied by 7 is 105." This complete loop — user message → model requests tool → you run tool → model gives final answer — is the core of tool calling. Agents automate this loop entirely.

---

## Install Everything

```bash
pip install langchain langchain-google-genai langchain-chroma langchain-community python-dotenv pydantic requests
```

---

## How to Run

```bash
python langchain_revision_project.py
```

Make sure your `.env` file is in the same folder with `GEMINI_API_KEY=your_key_here`.
