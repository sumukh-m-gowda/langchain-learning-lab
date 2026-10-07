import os
from dotenv import load_dotenv
import screen_brightness_control as sbc
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def set_brightness(level: int) -> str:
    """Set screen brightness from 0 to 100."""
    level = max(0, min(100, level))
    sbc.set_brightness(level)
    return f"Brightness set to {level}%"

@tool
def open_settings() -> str:
    """Open the Windows Settings app."""
    os.startfile("ms-settings:")
    return "Settings opened"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent = create_react_agent(llm, [set_brightness, open_settings])

while True:
    text = input("You: ")
    if text.lower() in ("exit", "quit"):
        break
    result = agent.invoke({"messages": [("user", text)]})
    print("Agent:", result["messages"][-1].content)