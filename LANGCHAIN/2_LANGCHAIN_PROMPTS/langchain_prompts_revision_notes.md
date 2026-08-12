# LangChain Prompts – Complete Revision Notes

---

## Why Prompts Matter

A prompt is the instruction you send to a model. LangChain provides a structured way to build, reuse, and manage prompts through **Prompt Templates** instead of hardcoding strings. This makes prompts dynamic, reusable, and easy to maintain.

---

## Temperature

Before diving into prompts, it's important to understand `temperature`, which controls how creative or random the model's output is.

- `temperature=0` → deterministic, always the same answer
- `temperature=1` → balanced (default)
- `temperature=1.5` or higher → more creative and random

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
```

---

## Message Types

LangChain uses structured message objects when talking to chat models. There are three types:

- `SystemMessage` – sets the behaviour/persona of the AI (e.g. "You are a helpful assistant")
- `HumanMessage` – represents what the user says
- `AIMessage` – represents what the model replied

These allow you to construct a full conversation context before sending it to the model.

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
```

After getting the result, the AI's reply is appended back to the `messages` list as an `AIMessage`. This is how conversation history is maintained manually.

---

## Building a Chatbot with Memory (Chat History)

A proper chatbot needs to remember what was said earlier in the conversation. This is done by maintaining a `chat_history` list that grows with each turn. Every user message and AI reply is appended, and the full list is sent to the model on every call.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(chat_history)
```

How it works step by step:
1. The loop starts with a `SystemMessage` already in `chat_history`
2. Each user input is wrapped in a `HumanMessage` and appended
3. The full `chat_history` list (including all past turns) is sent to the model
4. The model's reply is wrapped in an `AIMessage` and appended
5. This repeats, so the model always has full context of the conversation
6. Typing `exit` breaks the loop and prints the full history

---

## Prompt Templates

Instead of writing prompt strings by hand every time, LangChain lets you define a template with placeholders `{like_this}` and fill them in later. This is the `PromptTemplate` class.

### PromptTemplate (for plain text / LLM-style prompts)

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# Fill the placeholder and create the prompt
prompt = template2.invoke({'name': 'nitish'})

result = model.invoke(prompt)

print(result.content)
```

`input_variables` declares all the placeholders in the template. When you call `.invoke()` with a dictionary, the placeholders are replaced with the provided values.

---

## Saving and Loading Prompt Templates

You can serialize a `PromptTemplate` to a JSON file so it can be reused across scripts or shared with others. This is done with `.save()` and `load_prompt()`.

### Saving a Template

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=['paper_input', 'style_input', 'length_input'],
    validate_template=True
)

template.save('template.json')
```

`validate_template=True` checks that all variables listed in `input_variables` actually appear as `{placeholders}` in the template string. The saved `template.json` looks like this:

```json
{
    "input_variables": ["length_input", "paper_input", "style_input"],
    "template": "\nPlease summarize the research paper titled \"{paper_input}\"...",
    "template_format": "f-string",
    "validate_template": true,
    "_type": "prompt"
}
```

### Loading a Template and Using It in a Streamlit App

`load_prompt()` reads the JSON back into a `PromptTemplate` object. The `|` operator (pipe) chains the template and model together – this is the LangChain Expression Language (LCEL). It means: fill the template, then send it to the model.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()
model = ChatOpenAI()

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beat GANs on Image Synthesis"
])

style_input = st.selectbox("Select Explanation Style", [
    "Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"
])

length_input = st.selectbox("Select Explanation Length", [
    "Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"
])

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
```

The `chain = template | model` line is key. The `|` pipe creates a chain where:
1. The template fills in the variables and produces a prompt
2. That prompt is automatically passed to the model
3. The model returns a result

Run this app with: `streamlit run prompt_ui.py`

---

## ChatPromptTemplate (for Chat Models)

`ChatPromptTemplate` is the chat-aware version of `PromptTemplate`. Instead of a single string, it holds a list of `(role, message)` tuples, each of which can have placeholders.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})

print(prompt)
```

Calling `.invoke()` returns a `ChatPromptValue` object containing the formatted messages. This can be passed directly to a chat model.

---

## MessagesPlaceholder (Injecting Chat History into a Template)

When building a customer support bot or any stateful chatbot, you need to insert a variable number of past messages into the middle of a `ChatPromptTemplate`. `MessagesPlaceholder` reserves a slot in the template for an entire list of messages.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# Load chat history from a file
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Create the prompt by injecting the history and the new query
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})

print(prompt)
```

The `chat_history.txt` file contains previous conversation turns:

```
HumanMessage(content="I want to request a refund for my order #12345.")
AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")
```

The template structure becomes: System → [all past messages injected here] → new Human query. This gives the model full context of the conversation when answering the new question.

---

## Summary of All Prompt Classes

| Class | Purpose | Key Method |
|---|---|---|
| `PromptTemplate` | Single-string prompt with placeholders | `.invoke({'var': 'value'})` |
| `ChatPromptTemplate` | Multi-role prompt with placeholders per role | `.invoke({'var': 'value'})` |
| `MessagesPlaceholder` | Inject a list of messages into a chat template | Pass list via `variable_name` key |
| `load_prompt()` | Load a saved template from a JSON file | `load_prompt('template.json')` |

---

## Key Concepts to Remember

- `temperature` controls randomness – higher means more creative output.
- Always use `SystemMessage` to set the persona of the AI at the start of a conversation.
- To give a chatbot memory, maintain a `chat_history` list and send the entire list to the model on every turn.
- `PromptTemplate` uses `{placeholders}` that are filled with a dictionary via `.invoke()`.
- `validate_template=True` ensures all declared `input_variables` are present as placeholders in the template string.
- `.save('template.json')` serializes a template; `load_prompt('template.json')` loads it back.
- The `|` pipe operator chains a template and a model together (LCEL – LangChain Expression Language): `chain = template | model`.
- `ChatPromptTemplate` takes a list of `(role, content)` tuples where both the role and content can have `{placeholders}`.
- `MessagesPlaceholder` is used when the number of messages to inject is dynamic (e.g. loading history from a file or database).
