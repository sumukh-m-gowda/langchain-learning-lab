from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
prompt = PromptTemplate(
    template = "write a title for catchy blog about {topic}.",
    input_variables=["topic"]
)
topic = input('enter a topic : ')
formatted_prompt = prompt.format(topic = topic)

blog_title = llm.invoke(formatted_prompt)

print("generated blog title" , blog_title)