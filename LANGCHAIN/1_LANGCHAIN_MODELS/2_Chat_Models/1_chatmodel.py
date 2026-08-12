# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# from dotenv import load_dotenv 

# load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

# result = model.invoke("who is christofer nolan director or dancer one word")

# result2 = model.invoke("list his one movie that he directed")

# print(result.content)
# print(result2.content)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0.7, #temperature can be used only in chat models
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = model.invoke("who is christofer nolan director or dancer one word")

print(result.content) # .content can ve onyl used in chat model