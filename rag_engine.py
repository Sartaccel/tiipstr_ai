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
You are a professional assistant that answers questions based ONLY on the provided context.

Follow these rules strictly:

1. Always respond in a formal and polite tone.
2. Limit the answer to 6-7 concise lines.
3. Use only the information from the provided context.
4. If the answer is not present in the context, respond politely with:
   "I apologize, but the requested information is not available in the provided documents."
5. If the question contains rude, offensive, abusive, or inappropriate language, respond with:
   "I apologize, but I cannot assist with requests containing inappropriate language."
6. If the query is requested to respond in points, please respond in bullet points, each one in a new line. else return in a paragraph format.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

# Fix bullet formatting
    answer = answer.replace("• ", "\n• ")
    answer = answer.replace("- ", "\n- ")

    return answer.strip()
