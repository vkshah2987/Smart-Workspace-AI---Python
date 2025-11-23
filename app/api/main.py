import os
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from pathlib import Path
from redis import Redis
from rq import Queue
from bson import ObjectId
from .storage import save_upload
from .clients.mongo_client import MongoClientWrapper
from .clients.faiss_client import faiss_search, upsert_vectors, delete_document
from .clients.gemini_client import embed_query, generate_answer, embed_texts
from .clients.reranker_client import rerank_candidates
from .schemas import UploadResponse, QueryRequest, QueryResponse, DocumentListResponse, DocumentInfo, DeleteResponse

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "app/uploads/")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("ingest", connection=redis_conn)

app = FastAPI(title="RAG Workspace AI (Mongo + FAISS)")

mongo = MongoClientWrapper(os.getenv("MONGO_URI", "mongodb://mongo:27017/"), os.getenv("MONGO_DB", "ragdb"))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(user_id: str, file: UploadFile = File(...)):
    try:
        saved_path = await save_upload(file, UPLOAD_DIR)
        doc = {
            "user_id": user_id,
            "filename": file.filename,
            "path": saved_path,
            "status": "queued"
        }
        res = mongo.insert_document(doc)
        doc_id = str(res.inserted_id)
        q.enqueue("worker.ingest_job", doc_id, user_id, saved_path)
        return UploadResponse(doc_id=doc_id, location=saved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        print("Reached till here 0")
        q_emb = embed_query(req.query_text)
        print("Reached till here 0.1")
        dense_hits = faiss_search(q_emb, top_k=10, user_id=req.user_id)
        print("Reached till here 0.2")
        sparse_hits = mongo.text_search(req.query_text, user_id=req.user_id, top_k=10)
        print("Reached till here 1")
        cand_map = {}

        for h in dense_hits + sparse_hits:
            cid = h["chunk_id"]
            if cid not in cand_map or h.get("score", 0) > cand_map[cid].get("score", 0):
                cand_map[cid] = h

        print("Reached till here 2")

        candidates = list(cand_map.values())
        ranked = rerank_candidates(req.query_text, candidates)
        top_k = ranked[:3]
        contexts = [c["text"] for c in top_k]

        print("Reached till here 3")

        answer = generate_answer(req.query_text, contexts)
        sources = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "score": c.get("score", None)} for c in top_k]

        print("Reached till here 4")

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{user_id}", response_model=DocumentListResponse)
async def list_documents(user_id: str):
    """List all documents for a specific user"""
    try:
        documents = mongo._sync_db.documents.find({"user_id": user_id})
        doc_list = []
        for doc in documents:
            doc_list.append(DocumentInfo(
                doc_id=str(doc["_id"]),
                filename=doc.get("filename", ""),
                status=doc.get("status", "unknown"),
                path=doc.get("path", "")
            ))
        return DocumentListResponse(user_id=user_id, documents=doc_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document_endpoint(doc_id: str):
    """Delete a document and all associated data (chunks, vectors, mappings)"""
    try:
        # Find the document first
        try:
            obj_id = ObjectId(doc_id)
        except:
            # If not a valid ObjectId, treat as string
            obj_id = doc_id
        
        doc = mongo._sync_db.documents.find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Delete chunks from MongoDB
        mongo._sync_db.chunks.delete_many({"doc_id": doc_id})
        
        # Delete vectors and mappings from FAISS service
        delete_document(doc_id)
        
        # Delete the document itself
        mongo._sync_db.documents.delete_one({"_id": obj_id})
        
        # Delete the physical file if it exists
        file_path = doc.get("path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        return DeleteResponse(
            doc_id=doc_id,
            message="Document and all associated data deleted successfully",
            deleted=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))