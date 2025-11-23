import os, requests
FAISS_URL = os.getenv("FAISS_SERVICE_URL", "http://faiss:8001")

def upsert_vectors(doc_id, chunks, embeddings, user_id):
    payload = {
        "doc_id": doc_id,
        "user_id": user_id,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "embedding": emb if isinstance(emb, list) else emb.tolist()
            } for c, emb in zip(chunks, embeddings)
        ]
    }
    r = requests.post(f"{FAISS_URL}/upsert", json=payload)
    r.raise_for_status()
    return r.json()

def faiss_search(query_embedding, top_k=10, user_id=None):
    payload = {
        "embedding": query_embedding,
        "top_k": top_k,
        "user_id": user_id
    }
    r = requests.post(f"{FAISS_URL}/search", json=payload)
    r.raise_for_status()
    return r.json()["results"]

def delete_document(doc_id):
    payload = {"doc_id": doc_id}
    r = requests.post(f"{FAISS_URL}/delete", json=payload)
    r.raise_for_status()
    return r.json()