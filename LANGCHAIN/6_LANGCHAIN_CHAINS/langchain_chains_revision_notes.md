# LangChain Chains – Complete Revision Notes

---

## What is a Chain?

A chain is multiple LangChain components connected together using the `|` pipe operator (LCEL – LangChain Expression Language). The output of each step automatically becomes the input of the next.

```
prompt | model | parser
```

There are four types of chains covered in this lecture:

1. **Simple Chain** – one prompt → one model → one parser
2. **Sequential Chain** – multiple steps one after another
3. **Parallel Chain** – multiple steps running at the same time
4. **Conditional Chain** – different steps run depending on the input

---

## Visualising Any Chain

Every chain in LangChain has a `.get_graph().print_ascii()` method that prints a visual diagram of the chain's structure in the terminal. Always useful for debugging and understanding the flow.

```python
chain.get_graph().print_ascii()
```

---

## 1. Simple Chain

The most basic chain. One prompt → one model → one parser. This is the foundation everything else builds on.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'cricket'})

print(result)

chain.get_graph().print_ascii()
```

Flow:
```
PromptTemplate → ChatOpenAI → StrOutputParser
```

`.invoke({'topic': 'cricket'})` fills the placeholder, sends to the model, and returns a plain string.

---

## 2. Sequential Chain

Multiple prompt-model-parser steps chained one after another. The output of step 1 becomes the input of step 2. This is possible because `StrOutputParser` converts the `AIMessage` into a plain string that the next `PromptTemplate` can accept.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()
```

Flow:
```
prompt1 → model → parser(string) → prompt2 → model → parser(string)
```

Why `StrOutputParser` is essential here: after step 1, the model returns an `AIMessage` object. `prompt2` expects a plain string for `{text}`. `StrOutputParser` bridges that gap by extracting `.content` automatically.

The output of `parser` (a plain string) gets passed directly into `prompt2` as the value for `{text}` because the key name in `prompt2` matches what's flowing through the chain.

---

## 3. Parallel Chain

`RunnableParallel` runs multiple chains **at the same time** (in parallel) and collects all their results into a single dictionary. This is faster than running them sequentially when the steps are independent of each other.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for
classification, regression and outliers detection...
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()
```

Flow:
```
             ┌──── prompt1 | model1 | parser ──── (notes)  ────┐
{text} ──────┤                                                   ├──── prompt3 | model1 | parser
             └──── prompt2 | model2 | parser ──── (quiz)   ────┘
```

How it works step by step:
1. `RunnableParallel` takes `{text}` and sends it to **both** inner chains simultaneously
2. One chain (using OpenAI) generates notes; the other (using Anthropic/Claude) generates a quiz
3. Both run at the same time — so total time ≈ time of the slower one, not the sum of both
4. `RunnableParallel` collects the results into a dict: `{'notes': '...', 'quiz': '...'}`
5. That dict is passed to `merge_chain` where `{notes}` and `{quiz}` are filled automatically from the dict keys
6. The final merged document is returned

Notice two different models are used — `ChatOpenAI` for notes and `ChatAnthropic` for quiz. You can mix any models in a parallel chain.

---

## 4. Conditional Chain (RunnableBranch)

`RunnableBranch` routes the input to **different chains** based on a condition. Think of it like an if-else statement inside a chain. Each branch is a `(condition, chain)` tuple. The last argument is the default (fallback) if no condition matches.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Classify sentiment using PydanticOutputParser
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# Step 2: Two different response chains based on sentiment
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Step 3: Branch based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()
```

Flow:
```
feedback → classifier_chain → Feedback(sentiment='positive')
                                        ↓
                              RunnableBranch
                              ├── if positive → prompt2 | model | parser
                              ├── if negative → prompt3 | model | parser
                              └── else        → "could not find sentiment"
```

How it works step by step:
1. `classifier_chain` takes the feedback text and classifies it → returns a `Feedback` Pydantic object with `.sentiment = 'positive'` or `'negative'`
2. That `Feedback` object is passed to `branch_chain`
3. `RunnableBranch` checks each condition (lambda function) one by one — first one that returns `True` wins
4. The matching chain runs and returns the appropriate response
5. If no condition matches, the default `RunnableLambda` returns the fallback string

`RunnableLambda` wraps a plain Python function so it can be used inside a chain as a step.

---

## New Runnables Introduced

| Runnable | What it does |
|---|---|
| `RunnableParallel` | Runs multiple chains simultaneously, returns a dict of all results |
| `RunnableBranch` | Routes to different chains based on conditions (if-else for chains) |
| `RunnableLambda` | Wraps a plain Python `lambda` or function so it can be used as a chain step |

---

## Summary of All Four Chain Types

| Chain Type | Class/Pattern | Use Case |
|---|---|---|
| Simple | `prompt \| model \| parser` | Single task, one prompt |
| Sequential | `prompt1 \| model \| parser \| prompt2 \| model \| parser` | Multi-step tasks where step 2 needs step 1's output |
| Parallel | `RunnableParallel({...})` | Independent tasks that can run at the same time |
| Conditional | `RunnableBranch(...)` | Different logic paths based on input content |

---

## Key Concepts to Remember

- The `|` pipe operator is LCEL — it chains any two LangChain Runnables together.
- `StrOutputParser` is what makes sequential chains possible — it converts `AIMessage` → plain string so the next `PromptTemplate` can accept it.
- `RunnableParallel` saves time when tasks are independent — both run simultaneously, total time = slowest task (not sum of all).
- The dict keys in `RunnableParallel({'notes': ..., 'quiz': ...})` must match the `input_variables` in the next prompt (`{notes}`, `{quiz}`).
- `RunnableBranch` checks conditions top to bottom — first match wins. Always add a default at the end.
- `RunnableLambda` lets you plug any plain Python function into a chain as a step.
- You can mix different model providers (OpenAI + Anthropic) in the same chain.
- `chain.get_graph().print_ascii()` prints a visual diagram of any chain — very useful for debugging.
- `partial_variables` is used for format instructions (fixed at template creation); `input_variables` is for user-supplied values (filled at `.invoke()` time).
