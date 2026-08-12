from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv 

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = llm.invoke("who is christofer nolan director or dancer one word")

result2 = llm.invoke("list his one movie that he directed")

print(result)
print(result2)

# as llm are really old we will work with chat-models