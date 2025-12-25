from backend.graphs.ingestion.state import RagIngestState


def embedd(state: RagIngestState) -> RagIngestState:
    """
    Converts text chunks into vector embeddings using a specified embedding model.
    """
    
    return state