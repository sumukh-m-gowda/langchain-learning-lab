# task - asking llm to generate detailed report on some topic and then to summarize the same report

# Without StrOutputParser (manual, verbose)



from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

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
    template="Write a 4 line summary on the 5 line . \n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic' : 'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})
result2 = model.invoke(prompt2)

print(result2.content)