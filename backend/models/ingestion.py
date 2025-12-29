from pydantic import BaseModel
from typing import Optional


class ApiIngestionResponse(BaseModel):
    status: str  # "success" or "error"
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    total_chunks: Optional[int] = None
    processing_time_seconds: float
    error_message: Optional[str] = None
