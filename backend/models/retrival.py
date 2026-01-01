from pydantic import BaseModel
from typing import Optional, List, Dict


class ApiRetrivalRequest(BaseModel):
    document_id:str
    query: str

class ApiRetrivalResponse(BaseModel):
    status: str  # "success" or "error"
    final_response: Optional[str] = None
    source_chunks: Optional[List[Dict]] = None
    web_results: Optional[List[Dict]] = None
    tool_calls_history: Optional[List[Dict]] = None
    conversation_messages: Optional[List[Dict]] = None
    error_message: Optional[str] = None