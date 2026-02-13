"""
Incrementally ingest THE-INDIAN-PENAL-CODE-1860.pdf into the existing ChromaDB.

This script APPENDS to the existing vector store — it does NOT reset or
overwrite the database.  Embedding model and chunking parameters are kept
identical to the original Constitution ingest so that all vectors remain
in the same semantic space.
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ────────────────────────────────────────────────────
PDF_PATH = "data/THE-INDIAN-PENAL-CODE-1860.pdf"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ── Safety checks ────────────────────────────────────────────────────
if not os.path.isfile(PDF_PATH):
    sys.exit(f"❌ PDF not found at '{PDF_PATH}'. Aborting.")

if not os.path.isdir(PERSIST_DIR):
    sys.exit(f"❌ Existing ChromaDB not found at '{PERSIST_DIR}'. Aborting.")

# ── 1. Load PDF ──────────────────────────────────────────────────────
print(f"⏳ Loading PDF from {PDF_PATH}...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages.")

# ── 2. Chunk ─────────────────────────────────────────────────────────
print("⏳ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "],
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks from the IPC.")

# ── 3. Embedding model (same as existing DB) ────────────────────────
print(f"⏳ Initializing Embedding Model ({EMBEDDING_MODEL})...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ── 4. Open existing DB and get count BEFORE adding ──────────────────
existing_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
)
count_before = existing_db._collection.count()
print(f"📊 Documents already in DB: {count_before}")

# ── 5. Add new chunks incrementally ─────────────────────────────────
print("⏳ Adding new documents to existing Vector Database...")
existing_db.add_documents(documents=chunks)

count_after = existing_db._collection.count()
added = count_after - count_before

print()
print("═" * 50)
print(f"🎉 Success!  Indian Penal Code (1860) ingested.")
print(f"   Chunks added : {added}")
print(f"   Total in DB  : {count_after}")
print("═" * 50)
