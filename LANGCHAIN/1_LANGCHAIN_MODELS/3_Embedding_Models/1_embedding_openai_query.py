from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    dimensions=32,
    google_api_key=os.getenv("GEMINI_API_KEY_embedding")
)

vector = embeddings.embed_query("Delhi is the capital of India")

print("Length:", len(vector))
print("First 10:", vector[:10])

