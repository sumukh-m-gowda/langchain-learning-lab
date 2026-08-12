# LangChain Structured Output – Complete Revision Notes

---

## Why Structured Output?

By default, a model returns a free-form text string. This is fine for chatbots but useless for applications that need to process the response programmatically (e.g. extract sentiment, build a table, populate a database). Structured output forces the model to return data in a predictable, typed format like a dictionary or a Python object.

LangChain's `.with_structured_output()` method is the main tool for this. You define a **schema** (the shape of the output you want) and pass it to the model. The model then guarantees its response matches that schema.

---

## Background: Pydantic

Before looking at how LangChain uses structured output, you need to understand **Pydantic**, because it is the most powerful way to define schemas in LangChain.

Pydantic is a Python library for data validation. You define a class that inherits from `BaseModel`, declare fields with type hints, and Pydantic automatically validates and coerces incoming data.

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

new_student = {'age': '32', 'email': 'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])  # prints 32 (int), not '32' (string) -- auto coerced

student_json = student.model_dump_json()
```

Key points about Pydantic:
- `Optional[int]` means the field can be an integer or `None`
- `Field(gt=0, lt=10)` adds validation constraints (greater than 0, less than 10)
- `Field(description=...)` adds a human-readable description – this is what the model reads to understand what to fill in
- Pydantic automatically **coerces** types: the string `'32'` passed for `age` becomes the integer `32`
- `dict(student)` converts the object to a plain Python dictionary
- `student.model_dump_json()` serialises the object to a JSON string

---

## Background: TypedDict

`TypedDict` is a lighter-weight alternative to Pydantic. It comes from Python's `typing` module and lets you define a dictionary with fixed keys and types. There is **no validation or coercion** – it is purely for type hinting.

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name': 'nitish', 'age': '35'}

print(new_person)  # {'name': 'nitish', 'age': '35'} -- '35' is NOT coerced to int
```

Notice that `'35'` stays as a string even though `age` is declared as `int`. TypedDict does not enforce types at runtime – it is just a hint.

---

## The Three Ways to Define a Schema for Structured Output

LangChain's `.with_structured_output()` accepts three kinds of schemas:

1. **Pydantic `BaseModel`** – full validation, returns a typed Python object
2. **`TypedDict`** – no validation, returns a plain Python dictionary
3. **JSON Schema (plain dict)** – raw JSON schema, returns a plain Python dictionary

All three use the same `.with_structured_output()` method. The difference is what you pass in and what you get back.

---

## Method 1: Pydantic Schema

Define a `BaseModel` class with fields, types, and `Field(description=...)`. The description tells the model what each field should contain. `Literal["pos", "neg"]` restricts the value to one of those two strings only. `Optional` means the field can be `None` if not found.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra...""")

print(result)
```

What `result` looks like:

```
Review(
    key_themes=['processor performance', 'camera quality', 'battery life', 'S-Pen', 'price'],
    summary='The Samsung Galaxy S24 Ultra is a powerful phone with an excellent camera...',
    sentiment='pos',
    pros=['Snapdragon 8 Gen 3 processor', '200MP camera', 'long battery life', 'S-Pen'],
    cons=['heavy and large', 'bloatware', 'expensive'],
    name='Nitish Singh'
)
```

Since a Pydantic object is returned, you access fields with dot notation: `result.sentiment`, `result.name`, `result.pros`.

---

## Method 2: TypedDict Schema

Same idea but uses `TypedDict` instead of `BaseModel`. Descriptions are added using `Annotated[type, "description"]` syntax since TypedDict has no `Field()`.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra...""")

print(result['name'])  # dictionary-style access
```

Since TypedDict returns a plain Python dict, you access fields with bracket notation: `result['name']`, `result['sentiment']`.

---

## Method 3: JSON Schema (Raw Dict)

If you don't want to use Python classes at all, you can pass a raw JSON schema dictionary directly. This is the most explicit format.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra...""")

print(result)
```

The `"required"` list tells the model which fields are mandatory. Fields not in `"required"` can be absent. Like TypedDict, this returns a plain Python dictionary.

You can also save the schema to a `.json` file and load it separately:

```json
{
    "title": "student",
    "description": "schema about students",
    "type": "object",
    "properties": {
        "name": "string",
        "age": "integer"
    },
    "required": ["name"]
}
```

---

## Using Structured Output with HuggingFace (Llama)

`.with_structured_output()` works with any LangChain-compatible model, not just OpenAI. Here it is used with a HuggingFace-hosted TinyLlama model. The schema definition is identical – only the model setup changes.

```python
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra...""")

print(result)
```

Note: smaller open-source models like TinyLlama may not follow structured output instructions as reliably as GPT-4 or Claude. For production use, larger models are recommended.

---

## Comparison of All Three Schema Methods

| Feature | Pydantic (`BaseModel`) | TypedDict | JSON Schema (dict) |
|---|---|---|---|
| Validation & coercion | Yes | No | No |
| Return type | Typed Python object | Plain dict | Plain dict |
| Field access | Dot notation (`result.name`) | Bracket notation (`result['name']`) | Bracket notation (`result['name']`) |
| Descriptions | `Field(description=...)` | `Annotated[type, "description"]` | `"description"` key in schema |
| Constraints (`gt`, `lt`) | Yes via `Field()` | No | Yes via JSON Schema keywords |
| Best for | Production, APIs, data pipelines | Simple extraction tasks | Language-agnostic schemas |

---

## Key Concepts to Remember

- `.with_structured_output(schema)` wraps a model and forces its output to match the schema you define.
- Pydantic is the most powerful option: it validates, coerces types, and returns a typed object with dot-notation access.
- TypedDict is simpler – no validation, returns a plain dict, descriptions go in `Annotated[type, "description"]`.
- JSON Schema is the raw format – useful for language-agnostic schemas or when loading from a file.
- `Literal["pos", "neg"]` restricts a field to a fixed set of allowed values – great for sentiment, categories, etc.
- `Optional[list[str]]` means the field can be a list of strings or `None` – use when the information may not always be present.
- `Field(description=...)` is critical – the model reads the description to understand what to put in each field.
- `"required"` in JSON Schema lists fields the model must always fill. Fields outside this list are optional.
- `.with_structured_output()` works with any LangChain-compatible chat model (OpenAI, Anthropic, HuggingFace, etc.).
