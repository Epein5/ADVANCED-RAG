from typing import List, Dict, Any
from backend.core.cross_encoder_client import get_cross_encoder


class RerankingService:
    '''
    Reranks the retrieved documents based on their relevance to the query using cross-encoder model.
    '''

    def __init__(self):
        """Initialize the reranking service with cross-encoder model."""
        self.cross_encoder = get_cross_encoder()
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Re-rank chunks based on their relevance to the query using cross-encoder.
        
        Args:
            query (str): The search query
            chunks (List[Dict]): List of chunk objects from RRF with properties like:
                - content: The chunk text
                - contextualized_chunk: The contextualized version of the chunk
                - source: Source document
                - page_number: Page number
                - etc.
            top_k (int): Number of top results to return (default: 10)
        
        Returns:
            List[Dict]: Top-k re-ranked chunks sorted by cross-encoder score
        """
        if not chunks:
            return []
        
        # Extract contextualized chunks for re-ranking
        chunk_texts = [chunk.get('contextualized_chunk', chunk.get('content', '')) for chunk in chunks]
        
        # Create pairs for cross-encoder
        pairs = [[query, text] for text in chunk_texts]
        
        # Get relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk['reranking_score'] = float(score)
        
        # Sort by reranking score and return top-k
        reranked = sorted(chunks, key=lambda x: x['reranking_score'], reverse=True)
        
        return reranked[:top_k]