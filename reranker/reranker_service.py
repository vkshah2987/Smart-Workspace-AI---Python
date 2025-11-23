from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class RerankReq(BaseModel):
    query: str
    candidates: list

@app.post("/rerank")
def rerank(req: RerankReq):
    pairs = [(req.query, c["text"]) for c in req.candidates]
    scores = model.predict(pairs)
    for c, s in zip(req.candidates, scores):
        c["score"] = float(s)
    ranked = sorted(req.candidates, key=lambda x: x["score"], reverse=True)
    return {"ranked": ranked}