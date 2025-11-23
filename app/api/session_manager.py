"""
Session Manager for conversational RAG system.
Handles session creation, conversation history, and context management.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict
from pymongo.database import Database
from bson import ObjectId


class SessionManager:
    """Manages user sessions and conversation history."""
    
    def __init__(self, db: Database):
        self.db = db
        self.sessions = db.sessions
        # Create indexes for efficient queries
        self.sessions.create_index([("user_id", 1), ("created_at", -1)])
        self.sessions.create_index("session_id", unique=True)
    
    def create_session(self, user_id: str, initial_query: Optional[str] = None) -> str:
        """
        Create a new session for a user.
        
        Args:
            user_id: The user identifier
            initial_query: Optional first query to start the session
            
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        session_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "conversation_history": [],
            "metadata": {
                "total_queries": 0,
                "document_references": []  # Track which documents were queried
            }
        }
        
        if initial_query:
            session_doc["conversation_history"].append({
                "timestamp": datetime.utcnow(),
                "role": "user",
                "content": initial_query
            })
            session_doc["metadata"]["total_queries"] = 1
        
        self.sessions.insert_one(session_doc)
        return session_id
    
    def get_session(self, session_id: str, user_id: str) -> Optional[Dict]:
        """
        Retrieve a session by ID, ensuring it belongs to the user.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier (for security)
            
        Returns:
            Session document or None if not found
        """
        return self.sessions.find_one({
            "session_id": session_id,
            "user_id": user_id
        })
    
    def add_message(
        self, 
        session_id: str, 
        user_id: str, 
        role: str, 
        content: str,
        sources: Optional[List[Dict]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            sources: Optional source citations for assistant responses
            doc_ids: Optional list of document IDs referenced in this turn
            
        Returns:
            True if successful, False otherwise
        """
        message = {
            "timestamp": datetime.utcnow(),
            "role": role,
            "content": content
        }
        
        if sources:
            message["sources"] = sources
        
        update_doc = {
            "$push": {"conversation_history": message},
            "$set": {"updated_at": datetime.utcnow()}
        }
        
        # Increment query count for user messages
        if role == "user":
            update_doc["$inc"] = {"metadata.total_queries": 1}
        
        # Track document references
        if doc_ids:
            update_doc["$addToSet"] = {
                "metadata.document_references": {"$each": doc_ids}
            }
        
        result = self.sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            update_doc
        )
        
        return result.modified_count > 0
    
    def get_conversation_history(
        self, 
        session_id: str, 
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            limit: Optional limit on number of recent messages
            
        Returns:
            List of conversation messages
        """
        session = self.get_session(session_id, user_id)
        if not session:
            return []
        
        history = session.get("conversation_history", [])
        
        if limit and limit > 0:
            history = history[-limit:]
        
        return history
    
    def list_user_sessions(
        self, 
        user_id: str, 
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict]:
        """
        List all sessions for a user.
        
        Args:
            user_id: The user identifier
            limit: Maximum number of sessions to return
            skip: Number of sessions to skip (pagination)
            
        Returns:
            List of session summaries
        """
        cursor = self.sessions.find(
            {"user_id": user_id},
            {
                "session_id": 1,
                "created_at": 1,
                "updated_at": 1,
                "metadata": 1,
                "conversation_history": {"$slice": 1}  # Get first message for preview
            }
        ).sort("updated_at", -1).skip(skip).limit(limit)
        
        sessions = []
        for session in cursor:
            summary = {
                "session_id": session["session_id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "total_queries": session.get("metadata", {}).get("total_queries", 0),
                "document_count": len(session.get("metadata", {}).get("document_references", [])),
                "preview": None
            }
            
            # Add preview from first user message
            history = session.get("conversation_history", [])
            if history:
                first_msg = history[0]
                if first_msg.get("role") == "user":
                    preview_text = first_msg.get("content", "")
                    summary["preview"] = preview_text[:100] + "..." if len(preview_text) > 100 else preview_text
            
            sessions.append(summary)
        
        return sessions
    
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier (for security)
            
        Returns:
            True if deleted, False otherwise
        """
        result = self.sessions.delete_one({
            "session_id": session_id,
            "user_id": user_id
        })
        return result.deleted_count > 0
    
    def build_context_prompt(
        self, 
        session_id: str, 
        user_id: str,
        max_history: int = 5
    ) -> str:
        """
        Build conversation context for LLM prompt.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            max_history: Maximum number of previous turns to include
            
        Returns:
            Formatted conversation history string
        """
        history = self.get_conversation_history(session_id, user_id, limit=max_history * 2)
        
        if not history:
            return ""
        
        context_lines = ["PREVIOUS CONVERSATION:"]
        for msg in history:
            role = msg["role"].upper()
            content = msg["content"]
            context_lines.append(f"{role}: {content}")
        
        context_lines.append("")  # Empty line separator
        return "\n".join(context_lines)
    
    def update_session_metadata(
        self, 
        session_id: str, 
        user_id: str, 
        metadata_updates: Dict
    ) -> bool:
        """
        Update session metadata.
        
        Args:
            session_id: The session identifier
            user_id: The user identifier
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            True if successful, False otherwise
        """
        result = self.sessions.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$set": {f"metadata.{k}": v for k, v in metadata_updates.items()}}
        )
        return result.modified_count > 0
