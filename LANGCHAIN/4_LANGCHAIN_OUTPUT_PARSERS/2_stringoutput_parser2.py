from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
#report 
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

#summary 
template2 = PromptTemplate(
    template="mention all the milestones created by him in order. \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'virat legacy'})

print(result)