import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

os.environ["GEMINI_API_KEY"] = "AIzaSyDtvouGZSTcpXcMhigxtOBkpshh8eyN1Cc"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("API Key ausente! Configure GEMINI_API_KEY.")

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve_chunks(question, k=3):
    question_vec = embedding_model.encode([question])
    _, indices = index.search(np.array(question_vec), k)
    return [chunks[i] for i in indices[0]]

def generate_answer(question):
    context = "\n\n".join(retrieve_chunks(question))
    prompt = f"Com base no conteúdo abaixo, responda à pergunta:\n\n{context}\n\nPergunta: {question}"
    response = llm.generate_content(prompt)
    return response.text

print("Faça perguntas com base nos documentos (digite 'sair' para encerrar)")
while True:
    q = input("Pergunta: ")
    if q.lower() in ["sair", "exit", "quit"]:
        print("Sessão encerrada.")
        break
    answer = generate_answer(q)
    print(f"Resposta:\n{answer}\n")
