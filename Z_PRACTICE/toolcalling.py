# from langchain_core.tools import tool
# from langchain_core.messages import HumanMessage, ToolMessage

# # ---------- 1. Define tools (the "menu") ----------
# @tool
# def wikipedia_lookup(topic: str) -> str:
#     """Look up stable, well-documented facts about a topic."""
#     return f"Wikipedia: The Great Wall of China is about 21,196 km long."

# @tool
# def web_search(query: str) -> str:
#     """Search the web for news, interviews, and recent information."""
#     return "News: Astronaut Yang Liwei said he could not see the Great Wall from orbit."

# @tool
# def calculator(expression: str) -> str:
#     """Evaluate a math expression, e.g. '21196 * 0.621'."""
#     return str(eval(expression))   # fine for a demo, never use eval on real user input


# tools = [wikipedia_lookup, web_search, calculator]
# tools_by_name = {t.name: t for t in tools}

# # ---------- 2. Give the menu to the LLM ----------
# llm_with_tools = llm.bind_tools(tools)      # llm = your Gemini / OpenAI model

# # ---------- 3. Ask a question ----------
# messages = [HumanMessage("What did astronauts say about seeing the Great Wall from space?")]

# response = llm_with_tools.invoke(messages)  # LLM picks a tool (writes the order slip)
# messages.append(response)
# print("LLM chose:", response.tool_calls)

# # ---------- 4. Your code runs the chosen tool (the kitchen) ----------
# for call in response.tool_calls:
#     result = tools_by_name[call["name"]].invoke(call["args"])
#     messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

# # ---------- 5. Send results back, LLM writes the final answer ----------
# final = llm_with_tools.invoke(messages)
# print("Final answer:", final.content)

