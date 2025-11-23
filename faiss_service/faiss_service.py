from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import hashlib
import json
from pymongo import MongoClient
import os
from pathlib import Path

app = FastAPI()
EMBED_DIM = os.getenv("EMBED_DIM")
INDEX_FILE = Path("/data/faiss.index")
INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

def reset_index(target_dim: int):
    global index
    INDEX_FILE.unlink(missing_ok=True)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(target_dim))
    print(f"[FAISS] Initialized index with dimension {target_dim}")

def load_index():
    if INDEX_FILE.exists():
        loaded = faiss.read_index(str(INDEX_FILE))
        configured_dim = int(EMBED_DIM) if EMBED_DIM else loaded.d
        if loaded.d != configured_dim:
            print(f"[FAISS] Stored index dimension {loaded.d} mismatches configured {configured_dim}; reinitializing empty index")
            reset_index(configured_dim)
            return index
        print(f"[FAISS] Loaded existing index with dimension {loaded.d}")
        return loaded
    dim = int(EMBED_DIM) if EMBED_DIM else 0
    if dim <= 0:
        # dimension inferred on first upsert/search
        print("[FAISS] No embedding dimension configured; awaiting first embedding to initialize index")
        return None
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))

index = load_index()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ragdb")
mongo = MongoClient(MONGO_URI)[MONGO_DB]

def ensure_index_dim(target_dim: int, allow_reset: bool = False):
    global index
    if target_dim <= 0:
        raise HTTPException(status_code=400, detail="Embedding vector dimension must be positive")

    configured_dim = int(EMBED_DIM) if EMBED_DIM else target_dim

    if configured_dim != target_dim:
        print(f"[FAISS] Incoming embedding dim {target_dim} overrides configured dim {configured_dim}")
        configured_dim = target_dim

    if index is None:
        reset_index(configured_dim)
        return

    if index.d == target_dim:
        return

    if allow_reset or index.ntotal == 0:
        print(f"[FAISS] Resetting index from dim {index.d} to {target_dim}")
        reset_index(target_dim)
        return

    raise HTTPException(status_code=500, detail=f"FAISS index dimension {index.d} mismatches embedding length {target_dim}; drop /data/faiss.index and reingest.")

def chunkid_to_int64(chunk_id: str) -> int:
    h = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=True)

@app.post("/upsert")
def upsert(payload: dict):
    chunks = payload["chunks"]
    ids = []
    vecs = []
    mappings = []
    for c in chunks:
        cid = c["chunk_id"]
        emb = np.array(c["embedding"], dtype="float32")
        ensure_index_dim(emb.shape[0], allow_reset=True)
        faiss.normalize_L2(emb.reshape(1, -1))
        vecs.append(emb)
        chunk_int_id = chunkid_to_int64(cid)
        ids.append(chunk_int_id)
        mappings.append({
            "_id": chunk_int_id,
            "chunk_id": cid,
            "doc_id": payload["doc_id"],
            "user_id": payload.get("user_id")
        })
    
    # Batch insert/update mappings
    for mapping in mappings:
        mongo.faiss_mappings.replace_one(
            {"_id": mapping["_id"]},
            mapping,
            upsert=True
        )
    
    vecs_np = np.vstack(vecs)
    index.add_with_ids(vecs_np, np.array(ids, dtype="int64"))
    faiss.write_index(index, str(INDEX_FILE))
    return {"ok": True, "inserted": len(chunks)}

class SearchReq(BaseModel):
    embedding: list
    top_k: int = 10
    user_id: str = None

@app.post("/search")
def search(req: SearchReq):
    emb = np.array(req.embedding, dtype="float32").reshape(1, -1)
    ensure_index_dim(emb.shape[1], allow_reset=False)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, req.top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        mapping = mongo.faiss_mappings.find_one({"_id": int(idx)})
        if not mapping:
            continue
        if req.user_id and mapping.get("user_id") != req.user_id:
            continue

        chunk = mongo.chunks.find_one({"chunk_id": mapping["chunk_id"]})
        results.append({
            "chunk_id": mapping["chunk_id"],
            "doc_id": mapping["doc_id"],
            "text": chunk.get("text", "") if chunk else "",
            "score": float(score)
        })
    return {"results": results}

class DeleteReq(BaseModel):
    doc_id: str

@app.post("/delete")
def delete(req: DeleteReq):
    """Delete all vectors and mappings for a given document"""
    # Find all mappings for this doc_id
    mappings = list(mongo.faiss_mappings.find({"doc_id": req.doc_id}))
    
    if not mappings:
        return {"ok": True, "deleted": 0, "message": "No vectors found for this document"}
    
    # Collect IDs to remove from FAISS index
    ids_to_remove = np.array([m["_id"] for m in mappings], dtype="int64")
    
    # Remove from FAISS index
    if index is not None and index.ntotal > 0:
        index.remove_ids(ids_to_remove)
        faiss.write_index(index, str(INDEX_FILE))
    
    # Delete mappings from MongoDB
    result = mongo.faiss_mappings.delete_many({"doc_id": req.doc_id})
    
    return {"ok": True, "deleted": result.deleted_count, "message": f"Deleted {result.deleted_count} vectors"}