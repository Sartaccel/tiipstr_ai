import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "faiss_db"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

db = None
embeddings = None


# -----------------------------
# Load Embeddings (once)
# -----------------------------
def get_embeddings():
    global embeddings

    if embeddings is None:
        print("Loading embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded")

    return embeddings


# -----------------------------
# Load FAISS (once)
# -----------------------------
def get_db():
    global db

    if db is None:
        print("Loading FAISS database...")

        db = FAISS.load_local(
            DB_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        print("FAISS loaded successfully")

    return db


# -----------------------------
# Answer Question
# -----------------------------
def answer_question(question: str) -> str:

    db = get_db()

    docs = db.similarity_search(question, k=3)

    if not docs:
        return "I apologize, but the requested information is not available in the provided documents."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a professional assistant that answers questions based ONLY on the provided context.

Rules:
1. Respond politely and professionally.
2. Limit the response to 6–7 concise lines.
3. Use ONLY the information from the context.
4. If the answer is not present, respond:
"I apologize, but the requested information is not available in the provided documents."
5. If the user uses rude or abusive language respond:
"I apologize, but I cannot assist with requests containing inappropriate language."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception:
        return "Sorry, the AI service is currently unavailable. Please try again later."