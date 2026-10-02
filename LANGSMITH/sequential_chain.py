from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.7
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({"topic": "SQLite"}))