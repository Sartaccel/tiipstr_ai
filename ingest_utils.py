import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_DIR = "data"
DB_DIR = "faiss_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def load_or_create_db():
    if os.path.exists(DB_DIR):
        return FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

def ingest_pdf(pdf_path):
    print(f"📄 Ingesting: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    docs = splitter.split_documents(documents)

    db = load_or_create_db()
    if db:
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(DB_DIR)
    print("✅ FAISS index updated")
