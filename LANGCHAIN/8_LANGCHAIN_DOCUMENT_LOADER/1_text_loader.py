from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

# print(type(docs))           # <class 'list'>
# print(len(docs))            # 1  (whole file = one document)
# print(docs[0].page_content) # the poem text
# print(docs[0].metadata)     # {'source': 'cricket.txt'}

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))