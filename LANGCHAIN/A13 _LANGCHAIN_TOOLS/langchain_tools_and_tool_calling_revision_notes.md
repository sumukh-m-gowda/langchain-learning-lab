# LangChain Tools and Tool Calling – Complete Revision Notes

---

## What are Tools?

Until now, everything you've built has been about text — sending a prompt, getting a text response. But LLMs on their own cannot do things like search the web, run a calculation, call an API, or execute code. They can only generate text.

**Tools** are functions that you give to the model so it can actually *do* things in the real world. The model decides when a tool is needed, what arguments to pass to it, and uses the result to give a better answer. This is the foundation of AI Agents.

This lecture covers two notebooks — first how to define tools, then how the model actually calls them.

---

## Every Tool Has Three Properties

Every tool in LangChain — whether built-in or custom — exposes these three things:

```python
tool.name         # the name the model uses to identify and call it
tool.description  # what the tool does — the model reads this to decide when to use it
tool.args         # the input parameters the tool expects
```

The description is the most important of the three. The model decides whether to use a tool entirely based on its description. A vague description = the model won't know when to use it.

---

## Built-in Tools

LangChain comes with ready-made tools you can use directly without writing any code.

### DuckDuckGoSearchRun

Searches the web using DuckDuckGo and returns results as a string. Useful for answering questions about current events that the model doesn't know from training.

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

print(results)

print(search_tool.name)         # duckduckgo_search
print(search_tool.description)  # A wrapper around DuckDuckGo Search...
print(search_tool.args)         # {'query': {'type': 'string', ...}}
```

### ShellTool

Runs shell commands on the machine and returns the output. Use with caution — it has no safeguards by default.

```python
from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('ls')

print(results)  # lists files in the current directory
```

---

## Custom Tools — Three Methods

When you need a tool that doesn't exist as a built-in, you build your own. There are three ways to do this.

---

### Method 1: @tool Decorator (Simplest)

The quickest way. Write a normal Python function, add type hints, add a docstring, and decorate it with `@tool`. LangChain reads the function name as the tool name, the docstring as the description, and the type hints to build the args schema automatically.

The three steps shown in the notebook:

```python
# Step 1 - plain function
def multiply(a, b):
    """Multiply two numbers"""
    return a * b

# Step 2 - add type hints
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# Step 3 - add @tool decorator
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
```

Once decorated, invoke it like any other LangChain component:

```python
result = multiply.invoke({"a": 3, "b": 5})
print(result)  # 15

print(multiply.name)         # multiply
print(multiply.description)  # Multiply two numbers
print(multiply.args)         # {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}
```

You can also inspect the full Pydantic JSON schema the model receives:

```python
print(multiply.args_schema.model_json_schema())
# {'description': 'Multiply two numbers', 'properties': {'a': {...}, 'b': {...}}, 'required': ['a', 'b'], ...}
```

---

### Method 2: StructuredTool (More Control)

Use this when you want more control over the description and argument metadata. You define the input schema separately as a Pydantic `BaseModel`, write the function separately, then combine them with `StructuredTool.from_function()`.

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 3, 'b': 3})

print(result)                    # 9
print(multiply_tool.name)        # multiply
print(multiply_tool.description) # Multiply two numbers
print(multiply_tool.args)        # includes Field descriptions
```

The advantage over `@tool` is that `Field(description=...)` in `MultiplyInput` gives the model richer information about each argument — useful for complex tools where argument names alone aren't self-explanatory.

---

### Method 3: BaseTool Class (Most Control)

The most explicit approach. You subclass `BaseTool`, define the schema as a Pydantic model, and implement the `_run()` method. Use this when building complex tools that need internal state or custom behaviour.

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a': 3, 'b': 3})

print(result)                    # 9
print(multiply_tool.name)        # multiply
print(multiply_tool.description) # Multiply two numbers
print(multiply_tool.args)        # includes Field descriptions
```

`_run()` is the method LangChain calls internally when the tool is invoked. The underscore prefix means it is the internal implementation — you always call `.invoke()` from outside, never `._run()` directly.

---

## Toolkit

A Toolkit is just a class that groups related tools together and returns them via a `get_tools()` method. It is a simple organizational pattern — not a special LangChain class — just a clean way to bundle tools that belong together.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)
# add => Add two numbers
# multiply => Multiply two numbers
```

You then pass `tools` wherever a list of tools is needed (e.g. `llm.bind_tools(tools)`).

---

## Tool Calling — How the Model Uses Tools

Defining tools is only half the story. Tool calling is how you actually connect tools to the model so it can decide when to use them and execute them.

---

## Step 1: Bind Tools to the Model

`bind_tools()` tells the model which tools are available. The model now knows the name, description, and args of each tool, and can decide to call one instead of generating a text response.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

When you call `llm_with_tools.invoke('Hi how are you')`, the model responds normally with text since no tool is needed. When you call it with a math question, the model responds with a **tool call request** instead of a text answer.

---

## Step 2: The Tool Calling Loop (Manual)

Tool calling is not automatic — you have to run the tool yourself and feed the result back to the model. Here is the exact flow:

```python
# 1. User asks a question
query = HumanMessage('can you multiply 3 with 1000')
messages = [query]

# 2. Model responds with a tool call request (not a text answer)
result = llm_with_tools.invoke(messages)
messages.append(result)

# result.tool_calls looks like:
# [{'name': 'multiply', 'args': {'a': 3, 'b': 1000}, 'id': '...'}]

# 3. YOU run the tool and get the result
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)

# 4. Send everything back to the model — it now gives a final text answer
final_answer = llm_with_tools.invoke(messages).content
print(final_answer)  # "The result of multiplying 3 by 1000 is 3000."
```

The messages list at the end looks like:

```
[HumanMessage, AIMessage(tool_calls=[...]), ToolMessage(content='3000'), AIMessage(content='The result is 3000.')]
```

The model never runs the tool itself. It only says "please call this tool with these arguments." You call it, add the result to the conversation, and ask the model again. The model then sees the tool result and gives a final human-readable answer.

---

## Real Example: Currency Converter with InjectedToolArg

This example has two tools that depend on each other — `get_conversion_factor` fetches the exchange rate from an API, and `convert` multiplies a value by that rate. The `conversion_rate` in `convert` is marked as `InjectedToolArg` because it comes from the result of the first tool, not from the model.

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import json

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f'https://v6.exchangerate-api.com/v6/YOUR_API_KEY/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value * conversion_rate
```

`InjectedToolArg` means the model will NOT be asked to provide `conversion_rate` — the model only sees `base_currency_value` in the args. The `conversion_rate` is injected by your code from the result of the first tool. This is how you chain tool outputs into tool inputs.

### Running Two Dependent Tools

```python
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 INR to USD')]

# Model decides to call both tools
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# Run each tool call in order
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)

    if tool_call['name'] == 'convert':
        # inject the conversion_rate from tool 1 into tool 2's args
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

# Final answer
print(llm_with_tools.invoke(messages).content)
```

The model returned two tool calls at once — one for each tool. You loop through them, run them in order, manually pass the conversion rate from tool 1 into tool 2's args, and send everything back to the model for a final answer.

---

## Agent — Automating the Tool Calling Loop

The manual tool calling loop above works but is tedious. An **Agent** automates this loop entirely. You give it tools and a system prompt, and it keeps calling tools and feeding results back until it has a final answer — all automatically.

```python
from langchain.agents import create_agent

agent_executor = create_agent(
    model=llm,
    tools=[get_conversion_factor, convert, multiply],
    system_prompt="You must always use the provided tools for any arithmetic calculation. Never compute math yourself — always call the appropriate tool."
)

response = agent_executor.invoke({
    "messages": [{"role": "user", "content": "can you tell me what is 87 times 45"}]
})

print(response["messages"][-1].content)
# 87 multiplied by 45 is 3915.
```

The agent internally runs the same loop you wrote manually: invoke the model → check for tool calls → run the tools → append results → invoke again → repeat until no more tool calls → return final answer. The `system_prompt` guides the agent's behaviour — here it forces the model to always use tools for math instead of computing it internally.

---

## Comparison of Three Custom Tool Methods

| Method | How to define | Best for |
|---|---|---|
| `@tool` decorator | Function + type hints + docstring | Quick, simple tools |
| `StructuredTool` | Separate Pydantic schema + function | When you need rich arg descriptions |
| `BaseTool` class | Subclass with `_run()` method | Complex tools needing internal state |

---

## Key Concepts to Remember

A tool's `description` is what the model reads to decide when and whether to use it. A bad description = the model won't use the tool correctly.

`bind_tools([...])` makes the model aware of tools. Without this, the model cannot call any tool even if you define them.

The model never runs tools itself — it only requests a tool call with arguments. You run it, append the result as a `ToolMessage`, and send the conversation back to the model.

`InjectedToolArg` hides a parameter from the model — the model won't try to fill it. You inject it manually from another tool's output. Use this when tools depend on each other.

`result.tool_calls` is a list — the model can request multiple tool calls in a single response (as seen in the currency example where it called both tools at once).

An Agent automates the entire tool calling loop. It keeps running tools and feeding results back until it reaches a final answer with no more tool calls needed.

The `@tool` decorator reads the docstring as the description and the type hints as the args schema — both are required for the decorator to work properly.

Install with: `pip install langchain langchain-core langchain-community langchain-google-genai duckduckgo-search requests`
