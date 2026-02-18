import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "faiss_db"
MODEL_DIR = "models/mistral"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

def ask(question):
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY from the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    result = pipe(prompt)
    print(result[0]["generated_text"].split("Answer:")[-1])

while True:
    q = input("\n❓ Ask: ")
    if q.lower() == "exit":
        break
    ask(q)
