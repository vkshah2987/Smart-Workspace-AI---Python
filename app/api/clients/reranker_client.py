import os, requests
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8501/rerank")

def rerank_candidates(query, candidates):
    payload = {
        "query": query,
        "candidates": candidates
    }

    r = requests.post(RERANKER_URL, json=payload)
    r.raise_for_status()
    return r.json()["ranked"]