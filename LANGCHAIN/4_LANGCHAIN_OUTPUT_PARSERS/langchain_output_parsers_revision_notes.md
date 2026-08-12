# LangChain Output Parsers – Complete Revision Notes

---

## Why Output Parsers?

When a chat model responds, it returns a `AIMessage` object. To use that response further in your code (pass it to another prompt, parse it as JSON, convert it to a Python object), you need to extract and transform it. **Output Parsers** sit at the end of a chain and automatically convert the model's raw output into whatever format you need.

There are four main output parsers in LangChain:

1. `StrOutputParser` – extracts plain text from a message object
2. `JsonOutputParser` – parses the model's response as JSON into a Python dict
3. `StructuredOutputParser` – parses into a dict using named `ResponseSchema` fields
4. `PydanticOutputParser` – parses into a typed Pydantic object with validation

All parsers share a common pattern:
- Call `parser.get_format_instructions()` to get a string you inject into your prompt, telling the model how to format its output
- Add the parser at the end of a chain using the `|` pipe operator

---

## How the Chain Pattern Works (LCEL Recap)

The `|` pipe operator chains steps together. The output of each step becomes the input of the next:

```
template | model | parser
```

This means:
1. `template` fills in the placeholders and produces a prompt
2. `model` receives the prompt and returns an `AIMessage`
3. `parser` receives the `AIMessage` and transforms it into the final output

---

## 1. StrOutputParser

### The Problem It Solves

Without a parser, you have to manually call `.content` on the model result every time. In a multi-step chain, you also can't directly pipe a model's `AIMessage` output into a `PromptTemplate` because the template expects a plain string, not a message object.

### Without StrOutputParser (manual, verbose)

This works but requires you to manually extract `.content` and re-invoke at every step:

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})  # manually extracting .content

result1 = model.invoke(prompt2)

print(result1.content)  # manually extracting .content again
```

### With StrOutputParser (clean chain)

`StrOutputParser` extracts `.content` automatically, making it possible to chain multiple templates and model calls in one clean pipeline:

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
```

How this chain flows:
1. `template1` fills in `{topic}` → produces a prompt string
2. `model` generates a detailed report → returns an `AIMessage`
3. `parser` (StrOutputParser) extracts `.content` → plain string
4. `template2` uses that string to fill `{text}` → produces a new prompt
5. `model` generates a 5-line summary → returns an `AIMessage`
6. `parser` extracts `.content` again → final plain string

Without `StrOutputParser`, step 4 would fail because `template2` expects a string but would receive an `AIMessage` object.

---

## 2. JsonOutputParser

Used when you want the model to return structured data as a JSON object (Python dict). The parser injects format instructions into your prompt, telling the model to respond in valid JSON, then automatically parses the response.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
```

Key details:
- `parser.get_format_instructions()` returns a string like "Return your answer as a JSON object..." which gets baked into the prompt via `partial_variables`
- `partial_variables` is used for variables that are always the same (not user-supplied). Unlike `input_variables`, they are filled in at template creation time, not at `.invoke()` time
- The result is a Python dictionary you can work with directly

---

## 3. StructuredOutputParser

Used when you want the model to return a fixed set of named fields. You define each field as a `ResponseSchema` with a name and description, then build the parser from that list. Good for when you know exactly what keys you want in your output dictionary.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
```

The result is a plain Python dictionary with exactly the keys you defined:

```python
{
    'fact_1': 'Black holes are regions of spacetime where gravity is so strong...',
    'fact_2': 'The boundary of a black hole is called the event horizon...',
    'fact_3': 'Stephen Hawking proposed that black holes emit radiation...'
}
```

`StructuredOutputParser.from_response_schemas(schema)` builds the parser from your list of `ResponseSchema` objects. The parser's format instructions tell the model to respond in a markdown JSON block with exactly those keys.

---

## 4. PydanticOutputParser

The most powerful parser. Instead of returning a plain dict, it returns a fully validated **Pydantic object**. You define a `BaseModel` class with types and constraints, and the parser both instructs the model to fill those fields and validates the result.

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place': 'sri lankan'})

print(final_result)
```

The result is a `Person` object with dot-notation access:

```python
Person(name='Ashan Perera', age=27, city='Colombo')
```

You can access: `final_result.name`, `final_result.age`, `final_result.city`

The `Field(gt=18, ...)` constraint means Pydantic will raise an error if the model returns an age of 18 or below – real validation happens here, not just type hints.

---

## `partial_variables` vs `input_variables` – Important Distinction

| | `input_variables` | `partial_variables` |
|---|---|---|
| When filled | At `.invoke()` time (user supplies) | At template creation time (always fixed) |
| Example use | `{topic}`, `{place}` – changes per call | `{format_instruction}` – always the same |
| Passed via | `.invoke({'topic': 'black hole'})` | `partial_variables={'format_instruction': parser.get_format_instructions()}` |

Format instructions never change between calls, so they go in `partial_variables`. User inputs that change per call go in `input_variables`.

---

## Comparison of All Four Parsers

| Parser | Output Type | Best For | Needs Schema? |
|---|---|---|---|
| `StrOutputParser` | Plain string | Multi-step chains, simple text extraction | No |
| `JsonOutputParser` | Python dict | Any JSON-shaped output, flexible structure | No (model decides shape) |
| `StructuredOutputParser` | Python dict | Fixed named fields, simple structured output | Yes (`ResponseSchema` list) |
| `PydanticOutputParser` | Pydantic object | Typed output with validation and constraints | Yes (`BaseModel` class) |

---

## Key Concepts to Remember

- All parsers use `.get_format_instructions()` to generate a string that tells the model how to format its response. This string must be injected into the prompt.
- Always use `partial_variables` (not `input_variables`) for format instructions in a `PromptTemplate`.
- `StrOutputParser` is essential for chaining multiple prompt-model steps together – it bridges the gap between a model's `AIMessage` output and the next template's string input.
- `JsonOutputParser` is the most flexible – the model decides the exact shape of the JSON.
- `StructuredOutputParser` is good for simple, fixed key-value outputs defined with `ResponseSchema`.
- `PydanticOutputParser` is the most powerful – it gives you a typed object with runtime validation (e.g. `gt=18` will actually reject invalid values).
- All four parsers are drop-in at the end of a chain: `chain = template | model | parser`
- For `PydanticOutputParser`, pass the class itself (not an instance): `PydanticOutputParser(pydantic_object=Person)`
