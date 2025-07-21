from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def chunk_text(path, chunk_size=500):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text("extracted_text.txt")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

print("Indexing completed!")
