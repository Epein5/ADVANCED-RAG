import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject
from typing import Optional
from backend.core.config import config
from backend.core.db.weaviate_client import get_weaviate_client
from backend.graphs.ingestion.state import RagIngestState


class VectorStoreService:
    
    def __init__(self, client: Optional[weaviate.WeaviateClient] = None):
        self.client = client or get_weaviate_client()
        self.collection_name = config.weaviate_collection_name
        self._ensure_schema_exists()
    
    def _ensure_schema_exists(self) -> None:
        if self.client.collections.exists(self.collection_name):
            return
        
        self.client.collections.create(
            name=self.collection_name,
            vector_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="context", data_type=DataType.TEXT),
                Property(name="contextualized_chunk", data_type=DataType.TEXT),
                Property(name="breadcrumbs", data_type=DataType.TEXT_ARRAY),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="line_number", data_type=DataType.INT),
                Property(name="chunk_type", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="document_name", data_type=DataType.TEXT),
            ]
        )
    
    def store_embeddings(self, state: RagIngestState) -> str:
        if not state.get("chunks"):
            raise ValueError("No chunks to store")
        
        collection = self.client.collections.get(self.collection_name)
        
        objects = [
            DataObject(
                properties={
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content"),
                    "context": chunk.get("context"),
                    "contextualized_chunk": chunk.get("contextualized_chunk"),
                    "breadcrumbs": chunk.get("breadcrumbs"),
                    "page_number": chunk.get("page_number"),
                    "line_number": chunk.get("line_number"),
                    "chunk_type": chunk.get("chunk_type"),
                    "document_id": state.get("document_id"),
                    "document_name": state.get("document_name"),
                },
                vector=chunk.get("embedding")
            )
            for chunk in state["chunks"]
        ]
        
        collection.data.insert_many(objects)
        return state.get("document_id")