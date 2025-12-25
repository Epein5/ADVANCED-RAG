from typing import List, Optional, TypedDict, Union

class ChunkData(TypedDict):
    chunk_id: str
    content: str
    context: Optional[str] = None
    contexulized_chunk: Optional[str] = None
    embedding: Optional[List[float]]
    breadcrumbs: Optional[List[str]] = None
    page_number: Optional[int] = None
    line_number: Optional[int] = None
    chunk_type: Optional[str] = None

class RagIngestState(TypedDict):
    document_name: str
    raw_text: Optional[Union[str, dict]]
    document_id: str
    total_chunks: int
    chunks:List[ChunkData]
    
