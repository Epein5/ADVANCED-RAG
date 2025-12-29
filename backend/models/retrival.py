from pydantic import BaseModel
from typing import Optional


class ApiRetrivalRequest(BaseModel):
    document_id:str
    query: str

class ApiRetrivalResponse(BaseModel):
    status: str  # "success" or "error"
    final_response: Optional[str] = None
    source_chunks: Optional[dict] = None
    error_message: Optional[str] = None