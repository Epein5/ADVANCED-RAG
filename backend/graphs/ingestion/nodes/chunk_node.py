import uuid
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.graphs.ingestion.state import RagIngestState, ChunkData
from backend.core.config import config


def get_dynamic_chunk_size(text_length: int) -> int:
    """Dynamically determine chunk size based on document length."""
    # Target 10-20 chunks for optimal processing
    ideal_chunk_size = max(text_length // 15, config.min_chunk_size)
    return min(ideal_chunk_size, config.max_chunk_size)


def clean_chunk_text(text: str) -> str:
    """Clean chunk by removing extra whitespace while preserving structure."""
    # Remove spaces after/before newlines
    text = text.replace('\n ', '\n').replace(' \n', '\n')
    # Strip each line and remove empty lines
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    return text.strip()


def chunk_node(state: RagIngestState) -> RagIngestState:
    """Processes raw text or documents into smaller, manageable chunks."""
    raw_text = state["raw_text"]
    
    # Extract and merge text from list of dicts
    if isinstance(raw_text, list):
        merged_text = "\n".join([item.get("text", "") for item in raw_text])
        page_info = {i: item.get("page") for i, item in enumerate(raw_text)}
    else:
        merged_text = raw_text
        page_info = {0: None}
    
    # Get dynamic chunk size based on text length (characters)
    chunk_size = get_dynamic_chunk_size(len(merged_text))
    
    # Initialize splitter with no overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    split_chunks = splitter.split_text(merged_text)
    
    # Generate ChunkData objects
    chunks: List[ChunkData] = []
    current_pos = 0
    
    for chunk_text in split_chunks:
        # Calculate line number (count newlines from start to current position)
        line_number = merged_text[:current_pos].count('\n') + 1
        
        # Determine page number (approximate based on position in document)
        page_number = None
        if len(page_info) > 1:
            text_fraction = current_pos / len(merged_text)
            page_index = min(int(text_fraction * len(page_info)), len(page_info) - 1)
            page_number = page_info[page_index]
        elif len(page_info) == 1:
            page_number = page_info[0]
        
        # Create chunk data
        chunk_data: ChunkData = {
            "chunk_id": str(uuid.uuid4()),
            "content": clean_chunk_text(chunk_text),
            "context": None,
            "contexulized_chunk": None,
            "embedding": None,
            "breadcrumbs": None,
            "page_number": page_number,
            "line_number": line_number,
            "chunk_type": None
        }
        
        chunks.append(chunk_data)
        current_pos += len(chunk_text)
    
    # Update state
    state["chunks"] = chunks
    state["total_chunks"] = len(chunks)
    
    return state