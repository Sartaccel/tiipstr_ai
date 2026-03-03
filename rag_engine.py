import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
DB_DIR = "faiss_db"
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_db():
    return FAISS.load_local(
        DB_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def answer_question(question: str) -> str:
    db = get_db()

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY from the context below.make the answer a little more descriptive to explain a begginer kind of a person
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content