### includes memory in step 1[using chat history list] and then message to diff btw human and ai messages[using messages concept and chathistory as dictionary]


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage 
from dotenv import load_dotenv
import os

load_dotenv() 

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chat_history = [
    SystemMessage(content='you r helpfull AI')
]

while True :
    user_input = input('YOU : ')
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("OGGY : ", result.content)

print(chat_history)