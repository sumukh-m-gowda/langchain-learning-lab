# LangChain Text Splitters – Complete Revision Notes

---

## Why Text Splitters?

When you load a document (PDF, text file, webpage), the full content is often thousands of words long. You cannot dump the entire thing into a prompt because:

LLMs have a context window limit — there is a maximum amount of text they can process at once. Even if it fits, sending irrelevant content wastes tokens and money. And the model performs better when given focused, relevant text.

The solution is to split documents into smaller chunks and later retrieve only the relevant chunks for a given question. This is the foundation of RAG.

There are four types of text splitters covered in this lecture:

1. CharacterTextSplitter — split by length (character count)
2. RecursiveCharacterTextSplitter — split by text structure (paragraphs → sentences → words)
3. RecursiveCharacterTextSplitter.from_language() — split by code/markdown structure
4. SemanticChunker — split by meaning (using embeddings)

---

## Two Key Parameters (apply to all splitters)

Before looking at each splitter, understand these two parameters that appear everywhere.

chunk_size is the maximum number of characters allowed in a single chunk.

chunk_overlap is how many characters are repeated between two consecutive chunks. This ensures that if a sentence gets split across two chunks, the context is not completely lost — the next chunk starts slightly before where the last one ended.

```
chunk_overlap=0  →  [chunk1][chunk2][chunk3]
chunk_overlap=50 →  [chunk1][  chunk2  ][  chunk3  ]   ← 50 chars repeated at boundaries
```

---

## Two Methods: split_text() vs split_documents()

split_text(text) takes a plain string as input and returns a list of plain strings. Use this when you have raw text in your code.

split_documents(docs) takes a list of Document objects (from a loader) and returns a list of Document objects. Use this when you have loaded content from a file or PDF. This method preserves the page_content and metadata from the original documents.

---

## 1. CharacterTextSplitter (Length Based)

The simplest splitter. It splits purely by character count. When separator is set to an empty string, it does not care about words or sentences — it cuts at exactly chunk_size characters regardless of where a word or sentence ends.

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)
```

separator='' means split anywhere with no regard for word or sentence boundaries. chunk_size=200 means each chunk is at most 200 characters. The output result is still a list of Document objects, each with page_content and metadata. This is the bluntest splitter — good for quick testing but not ideal for real use because it can cut words in half.

---

## 2. RecursiveCharacterTextSplitter (Text Structure Based)

The most commonly used splitter in practice. Instead of cutting blindly at a character count, it tries to split at natural text boundaries in a specific priority order. It first tries to split at paragraph breaks. If a chunk is still too big after that, it tries line breaks. If still too big, it splits at spaces (word boundaries). If still too big after that, it splits at individual characters.

It only moves to the next level when the current level still produces chunks that are too large. This means it preserves the natural structure of text as much as possible.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)
```

split_text(text) is used here because the input is a plain string, not a loaded Document. The text has two paragraphs separated by a blank line — the splitter will first try to split there. Since each paragraph is under 500 characters, each paragraph becomes one chunk, giving 2 chunks total. This is the recommended default splitter for most use cases.

---

## 3. RecursiveCharacterTextSplitter.from_language() — Code and Markdown Splitting

When splitting code or markdown, plain text rules do not apply. You do not want to split in the middle of a function or cut a markdown section in half. from_language() is aware of the syntax of specific languages and splits only at meaningful boundaries for that language.

### Markdown Splitting

Splits at markdown headers and section boundaries, keeping each section together instead of cutting mid-paragraph.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo
   git clone https://github.com/your-username/student-tracker.git
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])
```

### Python Code Splitting

Splits at Python-specific boundaries like class definitions and function definitions, keeping logical code units together instead of splitting in the middle of a method.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])
```

Language.MARKDOWN and Language.PYTHON are enums — other options include Language.JS, Language.HTML, Language.SQL and more. The same chunk_size and chunk_overlap parameters apply. Use this when building a code documentation chatbot or a RAG system over a codebase.

---

## 4. SemanticChunker (Meaning Based)

All the splitters above split based on size or structure. SemanticChunker is different — it splits based on meaning. It uses embeddings to figure out where the topic actually changes in the text and places the split there, regardless of paragraph breaks or character count.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
```

The sample text above has three completely different topics mixed together — farming, cricket, and terrorism — without clean paragraph separation between the first two. SemanticChunker embeds each sentence and measures how much the meaning shifts between consecutive sentences. When the shift is large enough to cross the threshold, it places a split. The result is 3 chunks, one per topic, even though the text was not neatly separated.

breakpoint_threshold_type="standard_deviation" means the threshold for a split is calculated using the statistical standard deviation of embedding distances across all sentences. breakpoint_threshold_amount=3 controls sensitivity — a higher number means fewer splits, a lower number means more splits.

create_documents([sample]) takes a list of strings and returns Document objects. Note this is a different method name from the others which use split_text() or split_documents().

This splitter needs an OpenAI API key because it calls the embeddings model for every sentence. Use it when your document mixes multiple topics and you want chunks that are truly semantically coherent, like long research papers, reports, or books.

---

## Comparison of All Four Splitters

| Splitter | Splits by | Best for | Needs API key? |
|---|---|---|---|
| CharacterTextSplitter | Character count only | Quick testing, simple use | No |
| RecursiveCharacterTextSplitter | Text structure (paragraphs → sentences → words) | General purpose — most common choice | No |
| from_language(Language.X) | Code/markdown syntax boundaries | Code files, markdown docs | No |
| SemanticChunker | Meaning (embeddings) | Mixed-topic documents, research papers | Yes (OpenAI) |

---

## Key Concepts to Remember

Text splitters exist because LLMs have context window limits — you cannot feed a whole book into a prompt.

chunk_size is the max characters per chunk. chunk_overlap is the characters shared between neighbouring chunks to preserve context at boundaries.

split_text() takes a plain string. split_documents() takes a list of Document objects and preserves their metadata.

RecursiveCharacterTextSplitter is the go-to default for most use cases because it respects natural text boundaries.

from_language() is for code and markdown — it understands the syntax and splits at logical boundaries like function definitions and section headers.

SemanticChunker is the smartest but most expensive — it calls the embeddings API for every sentence to find where meaning changes.

Text splitting is step 2 of RAG. The full pipeline is: Load → Split → Embed → Store → Retrieve → Answer.

Install with: pip install langchain langchain-community langchain-experimental pypdf
