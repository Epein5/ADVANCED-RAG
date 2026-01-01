import json
from datetime import datetime
from typing import List, Dict, Optional
from backend.core.redis_client import get_redis_client
from backend.core.config import config


class ConversationService:
    """
    Service for managing conversation history in Redis.
    Stores and retrieves conversation messages per document.
    """
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.ttl_seconds = config.conversation_ttl_seconds
        self.max_history = config.max_conversation_history
    
    def _get_key(self, document_id: str) -> str:
        """Generate Redis key for conversation."""
        return f"conversation:{document_id}"
    
    def load_conversation(self, document_id: str) -> List[Dict]:
        """
        Load conversation history for a document.
        Returns last N messages (N from config.max_conversation_history).
        """
        messages_json = self.redis_client.get(self._get_key(document_id))
        if messages_json is None:
            return []
        
        try:
            messages = json.loads(messages_json)
            return messages[-self.max_history:]  # Naturally handles edge cases
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode conversation: {str(e)}")
    
    def save_conversation(self, document_id: str, messages: List[Dict]) -> None:
        """
        Save conversation history for a document.
        Stores full conversation with TTL from config.
        """
        try:
            self.redis_client.setex(
                self._get_key(document_id),
                self.ttl_seconds,
                json.dumps(messages)
            )
        except Exception as e:
            raise Exception(f"Failed to save conversation: {str(e)}")
    
    def clear_conversation(self, document_id: str) -> bool:
        """Delete conversation history for a document."""
        try:
            return self.redis_client.delete(self._get_key(document_id)) > 0
        except Exception as e:
            raise Exception(f"Failed to clear conversation: {str(e)}")
    
    def conversation_exists(self, document_id: str) -> bool:
        """Check if conversation exists for a document."""
        try:
            return self.redis_client.exists(self._get_key(document_id)) > 0
        except Exception as e:
            raise Exception(f"Failed to check conversation existence: {str(e)}")
