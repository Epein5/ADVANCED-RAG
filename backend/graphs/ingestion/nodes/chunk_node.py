import uuid
import logging
import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.graphs.ingestion.state import RagIngestState, ChunkData
from backend.core.config import config
from backend.utils.decorators import track_execution_time

logger = logging.getLogger(__name__)



def get_dynamic_chunk_size(text_length: int) -> int:
    """Dynamically determine chunk size based on document length."""
    # Target 10-20 chunks for optimal processing
    ideal_chunk_size = max(text_length // 15, config.min_chunk_size)
    return min(ideal_chunk_size, config.max_chunk_size)


def clean_chunk_text(text: str) -> str:
    """Clean chunk by removing extra whitespace while preserving paragraph structure."""
    # Replace multiple spaces with single spaces
    text = re.sub(r' +', ' ', text)
    # Remove excessive newlines but preserve paragraph breaks (double newlines)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Collapse multiple newlines to double
    # Clean up spaces around newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text.strip()


def merge_raw_text(raw_text) -> tuple[str, dict]:
    """Merge raw text from list of dicts or return as string."""
    if isinstance(raw_text, list):
        merged_text_parts = []
        page_info = {}
        
        for i, item in enumerate(raw_text):
            text = item.get("text", "")
            page_num = item.get("page")
            
            merged_text_parts.append(text)
            page_info[i] = page_num
        
        merged_text = "\n".join(merged_text_parts)
        return merged_text, page_info
    else:
        merged_text = raw_text
        page_info = {0: None}
        return merged_text, page_info


def create_chunk_data(chunk_text: str, merged_text: str, page_info: dict, raw_text: list, current_pos: int) -> ChunkData:
    """Create a ChunkData object for a given chunk."""
    # Find which input item this chunk belongs to
    item_index = 0
    cumulative_pos = 0
    
    for i, item in enumerate(raw_text):
        item_length = len(item.get("text", "")) + 1  # +1 for \n
        if current_pos < cumulative_pos + item_length:
            item_index = i
            break
        cumulative_pos += item_length
    
    # Get page and line info from the original item
    item = raw_text[item_index] if item_index < len(raw_text) else {}
    page_number = item.get("page")
    line_number = item.get("line_number")  # Only set for text files, None for PDFs
    
    return {
        "chunk_id": str(uuid.uuid4()),
        "content": clean_chunk_text(chunk_text),
        "context": None,
        "contextualized_chunk": None,
        "embedding": None,
        "breadcrumbs": None,
        "page_number": page_number,
        "line_number": line_number,
        "chunk_type": None
    }


def split_text_into_chunks(merged_text: str, chunk_size: int) -> list[str]:
    """Split merged text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(merged_text)


def log_chunking_summary(chunks: List[ChunkData], chunk_size: int, merged_text: str):
    """Log summary of chunking process."""
    chunk_lengths = [len(c["content"]) for c in chunks]
    if chunk_lengths:
        logger.info(
            "Chunking produced %d chunks — requested_chunk_size=%d, min=%d, max=%d, avg=%.1f",
            len(chunk_lengths),
            chunk_size,
            min(chunk_lengths),
            max(chunk_lengths),
            sum(chunk_lengths) / len(chunk_lengths),
        )
        logger.debug("Per-chunk sizes: %s", chunk_lengths)
    else:
        logger.warning("Chunking produced 0 chunks (input length=%d)", len(merged_text))

@track_execution_time
def chunk_node(state: RagIngestState) -> RagIngestState:
    """Processes raw text or documents into smaller, manageable chunks."""
    raw_text = state["raw_text"]
    
    # Merge raw text and extract page info
    merged_text, page_info = merge_raw_text(raw_text)
    
    # Get dynamic chunk size
    chunk_size = get_dynamic_chunk_size(len(merged_text))
    
    # Split text into chunks
    split_chunks = split_text_into_chunks(merged_text, chunk_size)
    
    # Generate ChunkData objects
    chunks: List[ChunkData] = []
    current_pos = 0
    
    for chunk_text in split_chunks:
        chunk_data = create_chunk_data(chunk_text, merged_text, page_info, raw_text, current_pos)
        chunks.append(chunk_data)
        current_pos += len(chunk_text)
    
    # Update state
    state["chunks"] = chunks
    state["total_chunks"] = len(chunks)
    
    # Log chunking summary
    log_chunking_summary(chunks, chunk_size, merged_text)
    
    return state