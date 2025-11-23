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
from .session_manager import SessionManager
from .schemas import (
    UploadResponse, QueryRequest, QueryResponse, DocumentListResponse, 
    DocumentInfo, DeleteResponse, SessionListResponse, SessionInfo,
    SessionDetail, SessionDeleteResponse, ConversationMessage, SessionMetadata
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "app/uploads/")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue("ingest", connection=redis_conn)

app = FastAPI(title="RAG Workspace AI (Mongo + FAISS)")

mongo = MongoClientWrapper(os.getenv("MONGO_URI", "mongodb://mongo:27017/"), os.getenv("MONGO_DB", "ragdb"))
session_manager = SessionManager(mongo._sync_db)

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
        # Handle session management
        session_id = req.session_id
        
        # Create new session if not provided
        if not session_id:
            session_id = session_manager.create_session(req.user_id, req.query_text)
            print(f"Created new session: {session_id}")
        else:
            # Validate existing session
            session = session_manager.get_session(session_id, req.user_id)
            if not session:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Session {session_id} not found or doesn't belong to user {req.user_id}"
                )
            # Add user message to existing session
            session_manager.add_message(session_id, req.user_id, "user", req.query_text)
            print(f"Continuing session: {session_id}")
        
        # Get conversation context for LLM
        conversation_context = session_manager.build_context_prompt(
            session_id, req.user_id, max_history=5
        )
        
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

        # Generate answer with conversation context
        answer = generate_answer(req.query_text, contexts, conversation_context)
        sources = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "score": c.get("score", None)} for c in top_k]

        print("Reached till here 4")
        
        # Extract unique document IDs for tracking
        doc_ids = list(set([s["doc_id"] for s in sources]))
        
        # Add assistant response to session
        session_manager.add_message(
            session_id, 
            req.user_id, 
            "assistant", 
            answer,
            sources=sources,
            doc_ids=doc_ids
        )

        return QueryResponse(answer=answer, sources=sources, session_id=session_id)

    except HTTPException:
        raise
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


# ===========================
# Session Management Endpoints
# ===========================

@app.get("/sessions/{user_id}", response_model=SessionListResponse)
async def list_sessions(user_id: str, limit: int = 50, skip: int = 0):
    """
    List all sessions for a user.
    
    Args:
        user_id: The user identifier
        limit: Maximum number of sessions to return (default: 50)
        skip: Number of sessions to skip for pagination (default: 0)
    """
    try:
        sessions = session_manager.list_user_sessions(user_id, limit=limit, skip=skip)
        
        # Get total count
        total = mongo._sync_db.sessions.count_documents({"user_id": user_id})
        
        session_infos = [
            SessionInfo(
                session_id=s["session_id"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                total_queries=s["total_queries"],
                document_count=s["document_count"],
                preview=s.get("preview")
            )
            for s in sessions
        ]
        
        return SessionListResponse(
            user_id=user_id,
            sessions=session_infos,
            total=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{user_id}/{session_id}", response_model=SessionDetail)
async def get_session(user_id: str, session_id: str):
    """
    Get detailed information about a specific session including full conversation history.
    
    Args:
        user_id: The user identifier
        session_id: The session identifier
    """
    try:
        session = session_manager.get_session(session_id, user_id)
        
        if not session:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {session_id} not found or doesn't belong to user {user_id}"
            )
        
        # Convert conversation history to schema format
        conversation = [
            ConversationMessage(
                timestamp=msg["timestamp"],
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources")
            )
            for msg in session.get("conversation_history", [])
        ]
        
        metadata = session.get("metadata", {})
        
        return SessionDetail(
            session_id=session["session_id"],
            user_id=session["user_id"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            conversation_history=conversation,
            metadata=SessionMetadata(
                total_queries=metadata.get("total_queries", 0),
                document_references=metadata.get("document_references", [])
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{user_id}/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(user_id: str, session_id: str):
    """
    Delete a session and all its conversation history.
    
    Args:
        user_id: The user identifier
        session_id: The session identifier
    """
    try:
        deleted = session_manager.delete_session(session_id, user_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {session_id} not found or doesn't belong to user {user_id}"
            )
        
        return SessionDeleteResponse(
            session_id=session_id,
            message="Session deleted successfully",
            deleted=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))