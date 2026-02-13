import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the raw text
print("⏳ Loading legal text...")
loader = TextLoader("data/bns_sample.txt", encoding='utf-8')
documents = loader.load()

# 2. Chunking (Cutting text into pieces)
# Legal texts are dense. We want small chunks so the AI doesn't get confused.
print("⏳ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Size of each piece
    chunk_overlap=50,     # Overlap ensures we don't cut a sentence in half
    separators=["\nSection", "\nCHAPTER", "\n", " "] # Try to split at logical points
)
chunks = text_splitter.split_documents(documents)
print(f"   - Created {len(chunks)} chunks from the law.")

# 3. Embedding (Text -> Numbers)
# We use a free, local model. (First run will download ~100MB model)
print("⏳ Initializing Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in VectorDB (Chroma)
print("⏳ Creating Vector Database...")
persist_directory = "./chroma_db"

# If DB exists, delete it to start fresh (for this tutorial)
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

print(f"✅ Knowledge Base built successfully! Saved to {persist_directory}")