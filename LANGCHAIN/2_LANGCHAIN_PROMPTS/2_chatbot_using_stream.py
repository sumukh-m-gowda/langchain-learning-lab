from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chat_history = [
    SystemMessage(content="you are a helpful AI")
]

while True:
    user_input = input("YOU : ")

    if user_input.lower() == "exit":
        break

    chat_history.append(
        HumanMessage(content=user_input)
    )

    print("OGGY : ", end="", flush=True)

    full_response = ""

    for chunk in model.stream(chat_history):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content

    print()

    chat_history.append(
        AIMessage(content=full_response)
    )

print(chat_history)