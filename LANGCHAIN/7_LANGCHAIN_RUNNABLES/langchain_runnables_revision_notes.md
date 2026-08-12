# LangChain Runnables – Complete Revision Notes

---

## What are Runnables?

Every component in LangChain (prompts, models, parsers, functions) is a **Runnable** — meaning it has a standard `.invoke()` method and can be connected with other Runnables using the `|` pipe operator.

This lecture covers the explicit Runnable classes that give you more control and clarity over how chains are built:

1. **RunnableSequence** – explicit version of the `|` pipe chain
2. **RunnableParallel** – run multiple chains at the same time
3. **RunnablePassthrough** – pass input forward without changing it
4. **RunnableLambda** – wrap a plain Python function as a Runnable
5. **RunnableBranch** – route to different chains based on a condition

---

## 1. RunnableSequence

`RunnableSequence` is the explicit, class-based way of writing what the `|` pipe does. Instead of `prompt | model | parser`, you pass all the steps as arguments to `RunnableSequence(...)`. Both are identical in behaviour — it's just two different syntaxes for the same thing.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic': 'AI'}))
```

This is a two-step sequential chain:
1. Generate a joke about the topic
2. Explain that joke

The output of `parser` (a plain string) automatically flows into `prompt2` as `{text}`.

**`|` pipe vs `RunnableSequence` — same thing:**
```python
# These two are identical
chain = prompt1 | model | parser | prompt2 | model | parser
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
```

---

## 2. RunnableParallel

`RunnableParallel` runs multiple chains simultaneously and returns a dictionary with all results. You saw this in the Chains lecture — this file shows a cleaner, simpler example.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'AI'})

print(result['tweet'])
print(result['linkedin'])
```

Both chains run at the same time with the same `{topic}` input. The result is a dict — access by key: `result['tweet']`, `result['linkedin']`.

---

## 3. RunnablePassthrough

`RunnablePassthrough` is a special Runnable that does **nothing** — it just passes whatever input it receives straight through unchanged. 

This sounds useless but is extremely powerful inside a `RunnableParallel` when you want to keep the original input alongside something new you've generated.

### The Problem it Solves

Imagine you generate a joke, then want to show both the joke **and** an explanation of it together. The problem: once the joke text flows into the explanation chain, the original joke text is gone. `RunnablePassthrough` preserves it.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic': 'cricket'}))
```

Flow:
```
{topic} → joke_gen_chain → "Why did the batsman..." (plain string)
                                      ↓
                          RunnableParallel receives the joke string
                          ├── 'joke': RunnablePassthrough()  → passes the joke string as-is
                          └── 'explanation': prompt2 | model | parser → generates explanation
                                      ↓
                          {'joke': 'Why did the batsman...', 'explanation': 'This joke works because...'}
```

`RunnablePassthrough()` inside the parallel chain receives the joke string and passes it forward unchanged into the `'joke'` key of the result dict.

The final output is:
```python
{
    'joke': 'Why did the batsman bring string to the match? To tie the score!',
    'explanation': 'This joke is a pun on the phrase "tie the score"...'
}
```

---

## 4. RunnableLambda

`RunnableLambda` wraps any plain Python function so it can be plugged into a chain as a step. This is how you add custom logic (that isn't a prompt, model, or parser) into a chain.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
```

Flow:
```
{topic} → joke_gen_chain → joke string
                                ↓
                    RunnableParallel receives joke string
                    ├── 'joke': RunnablePassthrough() → joke string as-is
                    └── 'word_count': RunnableLambda(word_count) → counts words → integer
                                ↓
                    {'joke': '...', 'word_count': 42}
```

`RunnableLambda(word_count)` takes the joke string, runs the `word_count` function on it, and returns the integer result. Without `RunnableLambda`, you can't plug a plain function into a chain directly.

The final printed output looks like:
```
Why don't AI models ever win at poker? Because they always show their hand!
word count - 17
```

---

## 5. RunnableBranch (with RunnablePassthrough as default)

This example combines `RunnableBranch` with `RunnablePassthrough` as the default fallback. If the report is short (under 300 words), there's no need to summarise — `RunnablePassthrough` just returns it as-is. If it's long (over 300 words), it gets summarised.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()
parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))
```

Flow:
```
{topic} → report_gen_chain → long report string
                                    ↓
                          RunnableBranch checks condition
                          ├── if word count > 300 → summarise it
                          └── else → RunnablePassthrough() → return as-is
```

`RunnablePassthrough()` as the default in `RunnableBranch` is a clean pattern — it means "if no condition matches, just return the input unchanged."

---

## All Runnables — Quick Reference

| Runnable | What it does | When to use |
|---|---|---|
| `RunnableSequence` | Chains steps one after another | Explicit alternative to `\|` pipe |
| `RunnableParallel` | Runs multiple chains simultaneously | Independent tasks, returns a dict |
| `RunnablePassthrough` | Passes input through unchanged | Preserve original value alongside new outputs |
| `RunnableLambda` | Wraps a plain Python function | Add custom logic (word count, formatting, etc.) |
| `RunnableBranch` | Routes to different chains based on condition | if-else logic inside a chain |

---

## Key Concepts to Remember

- `RunnableSequence(a, b, c)` and `a | b | c` are **identical** — two syntaxes for the same thing.
- `RunnablePassthrough` is most useful **inside `RunnableParallel`** to keep the original input alongside generated outputs.
- `RunnableLambda` is how you plug **any plain Python function** into a chain — the function receives the current value flowing through the chain and returns a new value.
- In `RunnableBranch`, `RunnablePassthrough()` as the default means "return input unchanged if no condition matches" — clean alternative to a fallback `RunnableLambda`.
- `RunnableParallel` both receives the **same input** for all its branches and returns a **single dict** with all results.
- The dict keys in `RunnableParallel` (`'tweet'`, `'linkedin'`, `'joke'`, `'word_count'`) are what you use to access results: `result['tweet']`.
