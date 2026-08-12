# LangChain Document Loaders – Complete Revision Notes

---

## What are Document Loaders?

Until now, you've been passing text directly into prompts as hardcoded strings. In real applications, your data lives in files — PDFs, text files, CSVs, websites. **Document Loaders** are LangChain's way of reading content from these sources and converting them into a standard format your chains can use.

Every loader returns a list of **Document objects**. A Document object has exactly two things:

- `page_content` — the actual text content
- `metadata` — information about where the content came from (file name, page number, source URL, etc.)

```python
docs[0].page_content   # the text
docs[0].metadata       # {'source': 'cricket.txt'} or {'source': 'file.pdf', 'page': 0}
```

There are five loaders covered in this lecture:

1. `TextLoader` – reads a `.txt` file
2. `PyPDFLoader` – reads a `.pdf` file (one Document per page)
3. `CSVLoader` – reads a `.csv` file (one Document per row)
4. `WebBaseLoader` – scrapes a webpage
5. `DirectoryLoader` – loads all files of a type from a folder

---

## 1. TextLoader

Reads a plain `.txt` file and returns it as a list with **one Document** (the entire file is one document).

```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))           # <class 'list'>
print(len(docs))            # 1  (whole file = one document)
print(docs[0].page_content) # the poem text
print(docs[0].metadata)     # {'source': 'cricket.txt'}

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))
```

Key points:
- `encoding='utf-8'` is important — without it, special characters in the file may cause errors
- `loader.load()` returns a list — even though there's only one document, you still access it as `docs[0]`
- `docs[0].page_content` gives the full text of the file, which is passed directly into the prompt as `{poem}`

---

## 2. PyPDFLoader

Reads a `.pdf` file and returns a list of Documents where **each page = one Document**.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))              # number of pages in the PDF
print(docs[0].page_content)   # text content of page 1
print(docs[1].metadata)       # {'source': 'dl-curriculum.pdf', 'page': 1}
```

Key points:
- A 20-page PDF → `len(docs)` = 20
- `docs[0]` = page 1, `docs[1]` = page 2, and so on
- `metadata` includes both the file name and the page number — very useful for citing sources later
- Install with: `pip install pypdf`

---

## 3. CSVLoader

Reads a `.csv` file and returns a list of Documents where **each row = one Document**.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))   # number of rows in the CSV
print(docs[1])     # second row as a Document object
```

What `docs[1]` looks like:

```
page_content='User ID: 15624510\nGender: Male\nAge: 19\nEstimatedSalary: 19000\nPurchased: 0'
metadata={'source': 'Social_Network_Ads.csv', 'row': 1}
```

Key points:
- Each row's column values are converted to `key: value` format inside `page_content`
- `metadata` includes the source file and the row number
- Useful for Q&A over tabular data — you can ask "which users purchased the product?"

---

## 4. WebBaseLoader

Scrapes a webpage from a URL and returns its text content as a Document. This lets you ask questions about any live webpage.

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({
    'question': 'What is the product that we are talking about?',
    'text': docs[0].page_content
}))
```

Key points:
- Pass a URL string to `WebBaseLoader(url)` — it fetches and scrapes the page automatically
- The entire page's text content is in `docs[0].page_content`
- The prompt here takes two variables: `{question}` (user's query) and `{text}` (the scraped page content)
- Install with: `pip install beautifulsoup4`
- This is the foundation of web-based Q&A — you can point it at any product page, news article, or documentation page and ask questions about it

---

## 5. DirectoryLoader

Loads **all files of a specific type** from a folder at once. Instead of loading files one by one, you point it at a directory and it handles everything.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
```

Key points:
- `path='books'` — the folder to look in
- `glob='*.pdf'` — the file pattern to match (`*.pdf` means all PDF files, `*.txt` would match all text files)
- `loader_cls=PyPDFLoader` — which loader to use for each file it finds
- `lazy_load()` instead of `load()` — loads files **one at a time** as you iterate over them instead of loading everything into memory at once. Useful when the folder has many large files.
- `load()` loads everything into memory immediately; `lazy_load()` is memory-efficient for large collections

---

## `.load()` vs `.lazy_load()`

| | `.load()` | `.lazy_load()` |
|---|---|---|
| Loads | Everything at once into a list | One document at a time as you loop |
| Memory | Higher (all in RAM) | Lower (one at a time) |
| Use when | Small files, need random access | Large folders, just iterating |

```python
# load() - everything at once
docs = loader.load()
print(docs[0])  # random access works

# lazy_load() - one at a time
for doc in loader.lazy_load():
    print(doc.metadata)  # process one by one
```

---

## The Document Object — Summary

Every loader, regardless of source, returns Documents with the same two fields:

```python
doc.page_content   # the text — this goes into your prompt
doc.metadata       # info about the source — useful for citations
```

Example metadata per loader:

| Loader | metadata example |
|---|---|
| TextLoader | `{'source': 'cricket.txt'}` |
| PyPDFLoader | `{'source': 'dl-curriculum.pdf', 'page': 0}` |
| CSVLoader | `{'source': 'Social_Network_Ads.csv', 'row': 1}` |
| WebBaseLoader | `{'source': 'https://flipkart.com/...'}` |
| DirectoryLoader | depends on `loader_cls` used |

---

## Required Packages

```bash
pip install langchain-community pypdf beautifulsoup4
```

---

## Key Concepts to Remember

- Document Loaders convert external files/URLs into LangChain `Document` objects with `page_content` and `metadata`.
- `TextLoader` → 1 file = 1 Document (whole file in one)
- `PyPDFLoader` → 1 file = N Documents (one per page)
- `CSVLoader` → 1 file = N Documents (one per row)
- `WebBaseLoader` → 1 URL = 1 Document (full scraped page)
- `DirectoryLoader` → 1 folder = all matching files loaded together
- Always use `docs[0].page_content` to get the text and pass it to your prompt.
- `lazy_load()` is memory-efficient — use it for large directories instead of `load()`.
- This topic is the **entry point to RAG** — in the next steps, instead of passing the whole document into a prompt, you'll split it into chunks, embed them, and retrieve only the relevant parts.
