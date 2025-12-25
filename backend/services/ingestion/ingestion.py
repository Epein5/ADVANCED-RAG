import time
import uuid
import asyncio
from typing import Optional

class DocumentIngestionService:
    '''
    Orchestrates the ingestion of documents:
    - loads raw text
    - runs processing graph
    - stores metadata and embeddings in weviate db
    '''
    def __init__(self, loader, graph, vector_store):
        self.loader = loader
        self.graph = graph
        self.vector_store = vector_store
        self.event_queue: Optional[asyncio.Queue] = None
    
    async def emit_event(self, status: str, message: str):
        """Emit progress event to queue"""
        if self.event_queue:
            event = f"data: {status}|{message}\n\n"
            await self.event_queue.put(event)
    
    def set_event_queue(self, queue: asyncio.Queue):
        """Set the event queue for streaming"""
        self.event_queue = queue
        
    async def ingest_document_async(self, file=None, text=None):
        """Async version with event streaming"""
        start_time = time.time()
        try:
            await self.emit_event("info", "Loading document...")
            raw_text = self.loader.load(file=file, text=text)
            document_name = file.filename if file else "raw_text_input"
            document_id = str(uuid.uuid4())
            
            await self.emit_event("info", "Starting chunking process...")
            graph_output = self.graph.invoke({
                "document_name": document_name,
                "document_id": document_id,
                "raw_text": raw_text,
                "total_chunks": 0,
                "chunks": []
            })
            
            await self.emit_event("chunks", f"Created {graph_output['total_chunks']} chunks")
            await self.emit_event("info", "Processing contextual retrieval...")
            # Graph completes all processing
            
            elapsed = time.time() - start_time
            
            result = {
                "status": "success",
                "document_id": graph_output["document_id"],
                "document_name": graph_output["document_name"],
                "total_chunks": graph_output["total_chunks"],
                "processing_time_seconds": elapsed
            }
            
            await self.emit_event("complete", "Ingestion completed successfully")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            await self.emit_event("error", str(e))
            raise
        
    def ingest_document(self, file=None, text=None):
        """Sync version (for backward compatibility)"""
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
            
            # document_id = self.vector_store.store_embeddings(
            #     graph_output
            # )
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