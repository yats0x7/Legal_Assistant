import ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup the Database Connection (The Memory)
print("⏳ Connecting to Legal Database...")
persist_directory = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def get_legal_answer(query):
    print(f"\n🔍 Analyzing Query: '{query}'")
    
    # 2. Retrieve Relevant Laws (The Research)
    results = vector_db.similarity_search(query, k=1) # Get top 1 most relevant section
    
    if not results:
        return "I could not find any specific Indian law regarding this in my database."
    
    context_text = results[0].page_content
    source_metadata = results[0].metadata
    print(f"   📄 Found Reference: {source_metadata.get('source', 'Legal Document')}")

    # 3. Construct the Prompt (The Brief)
    # We tell the LLM exactly how to behave.
    prompt = f"""
    You are an expert Indian Legal Assistant. 
    Use the following piece of "Context" (Indian Law) to answer the user's question.
    
    --- CONTEXT STARTS ---
    {context_text}
    --- CONTEXT ENDS ---
    
    Question: {query}
    
    Rules:
    1. Answer ONLY based on the context provided.
    2. Mention the specific Section Number if available.
    3. Keep the tone professional but easy to understand.
    4. If the answer is not in the context, say "I don't have enough information."
    """

    # 4. Generate Answer (The Drafting)
    print("   🤖 Drafting response with Llama 3...")
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

# --- Test Loop ---
if __name__ == "__main__":
    print("--- ⚖️  AI Lawyer Ready (Local Llama 3) ---")
    while True:
        user_input = input("\nAdmin: Ask a legal question (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        answer = get_legal_answer(user_input)
        print("\n📝 AI Response:")
        print(answer)