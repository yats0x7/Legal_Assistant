"""
One-time script to remove duplicate documents from ChromaDB.
Keeps the first occurrence of each unique chunk and deletes the rest.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"

print("⏳ Loading ChromaDB...")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed_model)

collection = db._collection
total = collection.count()
print(f"📊 Total documents before dedup: {total}")

# Fetch all documents (content + ids)
all_data = collection.get(include=["documents"])
ids = all_data["ids"]
docs = all_data["documents"]

# Find duplicate IDs — keep first occurrence, mark rest for deletion
seen = {}
ids_to_delete = []
for doc_id, content in zip(ids, docs):
    if content in seen:
        ids_to_delete.append(doc_id)
    else:
        seen[content] = doc_id

if ids_to_delete:
    print(f"🗑️  Found {len(ids_to_delete)} duplicates. Removing...")
    collection.delete(ids=ids_to_delete)
    print(f"✅ Removed {len(ids_to_delete)} duplicate chunks.")
else:
    print("✅ No duplicates found!")

print(f"📊 Total documents after dedup: {collection.count()}")
