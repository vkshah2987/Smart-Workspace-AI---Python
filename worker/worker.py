import os
from processors import process_document
from pymongo import MongoClient
from bson import ObjectId
from api.clients.gemini_client import embed_texts
from api.clients.faiss_client import upsert_vectors, faiss_search

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ragdb")

mongo = MongoClient(MONGO_URI)[MONGO_DB]

def ingest_job(doc_id, user_id, path):
    chunks = process_document(path, doc_id=doc_id)

    for c in chunks:
        mongo.chunks.insert_one({
            "chunk_id": c["chunk_id"],
            "doc_id": doc_id,
            "user_id": user_id,
            "text": c["text"],
            "seq": c["seq"],
            "tokens": c["tokens"]
        })

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_vectors(doc_id, chunks, embeddings, user_id=user_id)
    
    # Convert string doc_id back to ObjectId for MongoDB query
    try:
        obj_id = ObjectId(doc_id)
    except:
        obj_id = doc_id  # Fallback if already ObjectId or other format
    
    mongo.documents.update_one({"_id": obj_id}, {"$set": {"status": "indexed"}})