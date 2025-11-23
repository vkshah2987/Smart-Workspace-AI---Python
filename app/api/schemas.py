from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    doc_id: str
    location: str

class QueryRequest(BaseModel):
    user_id: str
    query_text: str

class Source(BaseModel):
    doc_id: str
    chunk_id: str
    score: Optional[float]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

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