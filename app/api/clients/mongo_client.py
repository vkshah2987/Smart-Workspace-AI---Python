from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import OperationFailure


def _build_text_filter(query: str, user_id: Optional[str]):
    base = {"$text": {"$search": query}}
    if user_id:
        base["user_id"] = user_id
    return base

class MongoClientWrapper:
    def __init__(self, uri, db_name):
        self._motor = AsyncIOMotorClient(uri)
        self.db = self._motor[db_name]
        self._sync_client = MongoClient(uri)
        self._sync_db = self._sync_client[db_name]

    def get_sync_client(self):
        return self._sync_client
    
    def insert_document(self, doc: dict):
        # FastAPI path requires sync insert, so delegate to PyMongo connection
        return self._sync_db.documents.insert_one(doc)

    def text_search(self, query_text: str, user_id: Optional[str] = None, top_k: int = 10):
        try:
            filter_query = _build_text_filter(query_text, user_id)
            projection = {
                "_id": 0,
                "chunk_id": 1,
                "doc_id": 1,
                "text": 1,
                "score": {"$meta": "textScore"}
            }
            cursor = self._sync_db.chunks.find(filter_query, projection).sort(
                [("score", {"$meta": "textScore"})]
            ).limit(top_k)
            results = list(cursor)
        except OperationFailure as exc:
            # Most common failure occurs when no text index exists; fall back to regex scan
            if "text index required" not in str(exc).lower():
                raise
            regex_filter = {"text": {"$regex": query_text, "$options": "i"}}
            if user_id:
                regex_filter["user_id"] = user_id
            projection = {
                "_id": 0,
                "chunk_id": 1,
                "doc_id": 1,
                "text": 1
            }
            cursor = self._sync_db.chunks.find(regex_filter, projection).limit(top_k)
            results = list(cursor)

        for doc in results:
            doc.setdefault("score", None)
        return results