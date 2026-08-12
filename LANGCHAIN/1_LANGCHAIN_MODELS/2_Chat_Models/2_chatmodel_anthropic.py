from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os 

load_dotenv() 

model = ChatAnthropic(model = 'model')

result = model.invoke("name of india captain")

print(result.content)