from backend.graphs.ingestion.state import RagIngestState


def contextual_retrival_node(state: RagIngestState) -> RagIngestState:
    """
    Applies Contextual Retrieval techniques to enhance chunk relevance and context.
    Also gathers metadata.
    """
    
    return state