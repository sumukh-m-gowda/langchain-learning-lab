from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    num_gpu=0
)

print(emb.embed_query("What is machine learning?"))