import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration
pdf_path = "data/coi-4March2016.pdf"
persist_directory = "./chroma_db"

print(f"⏳ Loading PDF from {pdf_path}...")
print("   (This might take a minute as the Constitution is over 400 pages long!)")

# 2. Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages.")

# 3. Chunking (Cutting into sections)
# We use larger chunks for the Constitution to capture full articles
print("⏳ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 1000 characters per chunk
    chunk_overlap=150,     # 150 characters overlap to prevent cutting sentences in half
    separators=["\n\n", "\n", ".", " "] 
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks from the Constitution.")

# 4. Embed and Store
print("⏳ Embedding and saving to Vector Database...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# We ADD to the existing database (we don't delete it this time)
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

print(f"🎉 Success! The Indian Constitution is now in your Assistant's brain.")