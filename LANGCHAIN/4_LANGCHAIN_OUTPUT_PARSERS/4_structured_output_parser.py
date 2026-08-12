## u can add a schema to this

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate ,
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers.structured import ResponseSchema
import os
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)

# {
#     'fact_1': 'Black holes are regions of spacetime where gravity is so strong...',
#     'fact_2': 'The boundary of a black hole is called the event horizon...',
#     'fact_3': 'Stephen Hawking proposed that black holes emit radiation...'
# }