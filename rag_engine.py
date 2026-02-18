# import torch
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# DB_DIR = "faiss_db"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

# @st.cache_resource
# def get_db():
#     return FAISS.load_local(
#         DB_DIR,
#         get_embeddings(),
#         allow_dangerous_deserialization=True
#     )

# @st.cache_resource
# def get_llm():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="auto",
#         load_in_4bit=True,
#         torch_dtype=torch.float16
#     )

#     return pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.2
#     )


# def answer_question(question: str) -> str:
#     db = get_db()
#     llm = get_llm()

#     docs = db.similarity_search(question, k=3)
#     context = "\n\n".join(d.page_content for d in docs)

#     prompt = f"""
# Answer ONLY from the context below.
# If the answer is not in the context, say "I don't know".

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     out = llm(prompt)
#     return out[0]["generated_text"].split("Answer:")[-1].strip()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "faiss_db"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load FAISS
def get_db():
    return FAISS.load_local(
        DB_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

# Load LLM (CPU safe)
def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )

    model.to("cpu")

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2
    )

# Load once globally
db = get_db()
llm = get_llm()

def answer_question(question: str) -> str:
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY from the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    out = llm(prompt)
    return out[0]["generated_text"].split("Answer:")[-1].strip()
