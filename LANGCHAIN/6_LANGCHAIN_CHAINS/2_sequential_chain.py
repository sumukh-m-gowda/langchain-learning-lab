# asking the llm to create a report on the topic and summarize the same report. here, we send the .invoke thing twice to model using chains .

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

prompt1 = PromptTemplate(
    template = 'generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate a 5 line summary on {report}',
    input_variables=['report']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 

result = chain.invoke({'topic':'unemplyment in india'})

print(result)