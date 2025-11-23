from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UploadResponse(BaseModel):
    doc_id: str
    location: str

class QueryRequest(BaseModel):
    user_id: str
    query_text: str
    session_id: Optional[str] = None  # Optional session ID for conversation continuity

class Source(BaseModel):
    doc_id: str
    chunk_id: str
    score: Optional[float]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str  # Always return session_id (new or existing)

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: str
    path: str

class DocumentListResponse(BaseModel):
    user_id: str
    documents: List[DocumentInfo]

class DeleteResponse(BaseModel):
    doc_id: str
    message: str
    deleted: bool

# Session-related schemas
class ConversationMessage(BaseModel):
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    content: str
    sources: Optional[List[Source]] = None

class SessionMetadata(BaseModel):
    total_queries: int
    document_references: List[str]

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    total_queries: int
    document_count: int
    preview: Optional[str] = None

class SessionDetail(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    conversation_history: List[ConversationMessage]
    metadata: SessionMetadata

class SessionListResponse(BaseModel):
    user_id: str
    sessions: List[SessionInfo]
    total: int

class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str
    deleted: bool