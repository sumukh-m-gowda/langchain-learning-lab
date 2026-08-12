# using the llm model only once in the chains

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = PromptTemplate(
    template = 'generate a 5 unbelievable stats on {player} ',
    input_variables=['player']
)
parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({'player' : 'virat kohli'})
print(result)