"""
ingest_json_qa.py
Ingests Q&A pairs from JSON files into the existing ChromaDB vector store.
"""
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_PATH = "./chroma_db"
DATA_DIR = "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    if not os.path.exists(filepath):
        print(f"   ⚠️  File not found: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_documents_from_indic_qa(data):
    """Create LangChain Documents from IndicLegalQA entries."""
    documents = []
    for item in data:
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        case_name = item.get('case_name', 'Unknown Case')
        judgment_date = item.get('judgment_date', 'Unknown Date')

        if not question or not answer:
            continue

        content = f"Q: {question}\nA: {answer}"
        metadata = {
            'source': 'IndicLegalQA_Dataset',
            'case_name': case_name,
            'judgment_date': judgment_date,
            'type': 'qa_pair'
        }
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def create_documents_from_constitution_qa(data):
    """Create LangChain Documents from Constitution Q&A entries."""
    documents = []
    for item in data:
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()

        if not question or not answer:
            continue

        content = f"Q: {question}\nA: {answer}"
        metadata = {
            'source': 'Constitution_QA',
            'type': 'qa_pair'
        }
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def ingest_in_batches(vector_db, documents, batch_size=BATCH_SIZE):
    """Add documents to ChromaDB in batches to avoid memory issues."""
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        vector_db.add_documents(batch)
        progress = min(i + batch_size, total)
        print(f"   📦 Ingested {progress}/{total} documents...")


def main():
    print("--- 📚 INGESTING JSON Q&A INTO CHROMADB ---\n")

    # 1. Load embedding model
    print("1️⃣  Loading embedding model...")
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 2. Connect to existing ChromaDB
    print("2️⃣  Connecting to ChromaDB...")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embed_model)

    # Check existing count
    existing_count = vector_db._collection.count()
    print(f"   📊 Existing documents in DB: {existing_count}")

    # 3. Load and process IndicLegalQA
    print("\n3️⃣  Processing IndicLegalQA Dataset...")
    indic_path = os.path.join(DATA_DIR, "IndicLegalQA Dataset_10K.json")
    indic_data = load_json_file(indic_path)
    indic_docs = create_documents_from_indic_qa(indic_data)
    print(f"   📄 Created {len(indic_docs)} documents from IndicLegalQA")

    # 4. Load and process Constitution Q&A
    print("\n4️⃣  Processing Constitution Q&A...")
    const_path = os.path.join(DATA_DIR, "constitution_qa.json")
    const_data = load_json_file(const_path)
    const_docs = create_documents_from_constitution_qa(const_data)
    print(f"   📄 Created {len(const_docs)} documents from Constitution Q&A")

    # 5. Combine and ingest
    all_docs = indic_docs + const_docs
    print(f"\n5️⃣  Ingesting {len(all_docs)} total documents into ChromaDB...")
    ingest_in_batches(vector_db, all_docs)

    # 6. Verify
    final_count = vector_db._collection.count()
    print(f"\n✅ INGESTION COMPLETE!")
    print(f"   📊 Documents before: {existing_count}")
    print(f"   📊 Documents after:  {final_count}")
    print(f"   📊 New documents:    {final_count - existing_count}")


if __name__ == "__main__":
    main()
