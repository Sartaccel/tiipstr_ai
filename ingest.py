import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_DIR = "data"
DB_DIR = "faiss_db"

documents = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        loader = PyMuPDFLoader(os.path.join(PDF_DIR, file))
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local(DB_DIR)

print("✅ PDFs indexed successfully")
