import time
import uuid

class DocumentIngestionService:
    '''
    Orchestrates the ingestion of documents:
    - loads raw text
    - runs processing graph
    - stores metadata and embeddings in weviate db
    '''
    def __init__(self,loader, graph, vector_store):
        self.loader = loader
        self.graph = graph
        self.vector_store = vector_store
        
    def ingest_document(self, file=None, text=None):
        start_time = time.time()
        try:
            raw_text = self.loader.load(file=file, text=text)
            document_name = file.filename if file else "raw_text_input"
            document_id = str(uuid.uuid4())
            
            graph_output = self.graph.invoke({
                "document_name": document_name,
                "document_id": document_id,
                "raw_text": raw_text,
                "total_chunks": 0,
                "chunks": []
            })
            
            document_id = self.vector_store.store_embeddings(
                graph_output
            )
            elapsed = time.time() - start_time

            return {
                "status": "success",
                "document_id": document_id,
                "document_name": file.filename if file else None,
                "total_chunks": len(graph_output["chunks"]),
                "processing_time_seconds": elapsed,
                "error_message": None
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "status": "error",
                "document_id": None,
                "document_name": file.filename if file else None,
                "total_chunks": None,
                "processing_time_seconds": elapsed,
                "error_message": str(e)
            }