# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os

# from rag_engine import answer_question

# # -----------------------------
# # App initialization
# # -----------------------------
# app = FastAPI(
#     title="Document-Aware RAG API",
#     description="Backend API for document-based question answering",
#     version="1.0.0"
# )

# # -----------------------------
# # CORS (allow frontend access)
# # -----------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change this in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # Constants
# # -----------------------------
# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # -----------------------------
# # Request / Response Models
# # -----------------------------
# class QuestionRequest(BaseModel):
#     question: str

# class AnswerResponse(BaseModel):
#     answer: str

# # -----------------------------
# # Health Check
# # -----------------------------
# @app.get("/health")
# def health_check():
#     return {"status": "ok"}

# # -----------------------------
# # Ask Question Endpoint
# # -----------------------------
# @app.post("/ask", response_model=AnswerResponse)
# def ask_question(payload: QuestionRequest):
#     if not payload.question.strip():
#         raise HTTPException(status_code=400, detail="Question cannot be empty")

#     answer = answer_question(payload.question)
#     return {"answer": answer}

# # -----------------------------
# # Upload PDF Endpoint
# # -----------------------------
# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed")

#     file_path = os.path.join(DATA_DIR, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     return {
#         "status": "uploaded",
#         "filename": file.filename,
#         "message": "File uploaded successfully. It will be indexed automatically."
#     }


from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from rag_engine import answer_question

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
API_SECRET = os.getenv("AI_API_SECRET")

if not API_SECRET:
    raise RuntimeError("AI_API_SECRET is not set in environment variables")

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="Document-Aware RAG API",
    description="Backend API for document-based question answering",
    version="1.0.0"
)

# -----------------------------
# CORS (Restrict in production)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Spring Boot origin
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# -----------------------------
# Security - API Key Verification
# -----------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized access")

# -----------------------------
# Constants
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Request / Response Models
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# -----------------------------
# Ask Question Endpoint (Secured)
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(
    payload: QuestionRequest,
    _: str = Depends(verify_api_key)
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = answer_question(payload.question)
    return {"answer": answer}

# -----------------------------
# Upload PDF Endpoint (Secured)
# -----------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "status": "uploaded",
        "filename": file.filename,
        "message": "File uploaded successfully. It will be indexed automatically."
    }
