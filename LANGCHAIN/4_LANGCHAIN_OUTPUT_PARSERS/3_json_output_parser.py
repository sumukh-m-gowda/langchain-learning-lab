from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = JsonOutputParser()

template = PromptTemplate(
    # template='give me the name , age and city of ronaldo \n {format_instruction}' ,
    template='give me the name , age and city of {topic} \n {format_instruction}' ,
    # input_variables=[],
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.format() 
# print(prompt) --- > give me the name , age and city of ronaldo  Return a JSON object. [this is the formate llm is recieving]
chain = template | model | parser

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# print(chain.invoke({}))
print(chain.invoke({'topic' : 'ronaldo(cr7)'}))