import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the Database
persist_directory = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def search_legal_db(query):
    print(f"\n🔎 Searching for: '{query}'")
    
    # K=2 means "Give me the top 2 most relevant sections"
    results = vector_db.similarity_search(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:300]}...") # Show first 300 chars
        print("-------------------")

if __name__ == "__main__":
    # Test queries
    search_legal_db("What is the punishment for murder?")
    search_legal_db("Can a man be punished for false promise of marriage?")