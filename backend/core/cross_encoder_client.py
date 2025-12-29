from sentence_transformers import CrossEncoder
from functools import lru_cache


@lru_cache(maxsize=1)
def get_cross_encoder():
    """
    Get or initialize the cross-encoder model for re-ranking.
    
    Uses LRU cache to ensure only one model instance is created and reused.
    
    Returns:
        CrossEncoder: The cross-encoder model for MS MARCO ranking
    """
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return model
