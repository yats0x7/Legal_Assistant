import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/"
DB_PATH = "./chroma_db"

def main():
    print(f"📂 Scanning '{DATA_PATH}' for all PDFs (including subfolders)...")
    
    # 1. Load ALL PDFs
    # glob="**/*.pdf" means "look in all folders for .pdf files"
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found! Check your data folder.")
        return

    print(f"✅ Loaded {len(documents)} pages from your library.")

    # 2. Split Text
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📊 Created {len(chunks)} text chunks.")

    # 3. Create/Update Vector Database
    print("🧠 Embedding data into the Knowledge Base (this takes time)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # This creates the DB if missing, or adds to it if it exists
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"🎉 Success! All data is now stored in {DB_PATH}")

if __name__ == "__main__":
    main()
    