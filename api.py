from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import threading
import rag_engine


# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------
load_dotenv()

API_SECRET = os.getenv("AI_API_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL")

if not API_SECRET:
    raise RuntimeError("AI_API_SECRET is not set in environment variables")


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Document Aware RAG API",
    description="API for document-based question answering using RAG",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # Frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------
# API Key Verification
# ---------------------------------------------------------
def verify_api_key(x_api_key: str = Header(None)):

    if x_api_key != API_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Unauthorized access"
        )


# ---------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():

    return {
        "status": "ok",
        "message": "RAG API is running"
    }


# ---------------------------------------------------------
# Ask Question Endpoint
# ---------------------------------------------------------
@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
def ask_question(
    payload: QuestionRequest,
    _: str = Depends(verify_api_key)
):

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        answer = rag_engine.answer_question(question)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI processing error: {str(e)}"
        )


# ---------------------------------------------------------
# Upload PDF Endpoint
# ---------------------------------------------------------
@app.post("/upload", tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(...),
    _: str = Depends(verify_api_key)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

    return {
        "status": "success",
        "filename": file.filename,
        "message": "PDF uploaded successfully"
    }


# ---------------------------------------------------------
# Load AI Resources at Startup
# ---------------------------------------------------------
@app.on_event("startup")
def load_ai_resources():

    print("🔥 Initializing AI resources...")

    try:
        threading.Thread(target=rag_engine.get_embeddings).start()
        threading.Thread(target=rag_engine.get_db).start()

        print("✅ AI resources loading in background")

    except Exception as e:
        print(f"❌ Failed to load AI resources: {e}")