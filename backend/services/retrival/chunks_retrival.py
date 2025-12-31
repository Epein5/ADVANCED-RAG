from backend.utils.decorators import track_execution_time
from weaviate.classes.query import Filter

class ChunksRetrivalService:
    '''
    Service to retrieve relevant chunks from a document based on a query.
    '''
    def __init__(self, weaviate_client, embedding_service):
        self.weaviate_client = weaviate_client
        self.embedding_service = embedding_service

    @track_execution_time
    def vector_search(self, query: str, document_id: str, top_k: int = 5) -> list[dict]:
        '''
        Perform a vector search to retrieve top_k relevant chunks for the given query and document_id.
        '''
        query_embedding = self.embedding_service.embedd_text(query)
        collection = self.weaviate_client.collections.get("Chunk")
        
        where_filter = Filter.by_property("document_id").equal(document_id)
        
        results = collection.query.near_vector(
            near_vector=query_embedding,
            filters=where_filter,
            limit=top_k
        )
        
        return [obj.properties for obj in results.objects]
    
    @track_execution_time
    def bm25_search(self, query: str, document_id: str, top_k: int = 5) -> list[dict]:
        '''
        Perform a BM25 search to retrieve top_k relevant chunks for the given query and document_id.
        '''
        collection = self.weaviate_client.collections.get("Chunk")
        
        where_filter = Filter.by_property("document_id").equal(document_id)
        
        results = collection.query.bm25(
            query=query,
            filters=where_filter,
            limit=top_k
        )
        
        return [obj.properties for obj in results.objects]
    

    @track_execution_time
    def reciprocal_rank_fusion(self, vector_results, bm25_results) -> list[dict]:
        '''
        Perform Reciprocal Rank Fusion (RRF) to combine vector and BM25 search results.
        '''
        score_dict = {}
        
        for rank, obj in enumerate(vector_results):
            chunk_id = obj["chunk_id"]
            score_dict[chunk_id] = score_dict.get(chunk_id, 0) + 1 / (rank + 1 + 60)
        
        for rank, obj in enumerate(bm25_results):
            chunk_id = obj["chunk_id"]
            score_dict[chunk_id] = score_dict.get(chunk_id, 0) + 1 / (rank + 1 + 60)
        
        sorted_chunk_ids = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        top_chunk_ids = {chunk_id for chunk_id, _ in sorted_chunk_ids}
        
        combined_results = {obj["chunk_id"]: obj for obj in vector_results + bm25_results if obj["chunk_id"] in top_chunk_ids}
        
        return [combined_results[chunk_id] for chunk_id, _ in sorted_chunk_ids]