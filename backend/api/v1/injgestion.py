from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import json

from backend.models.ingestion import ApiIngestionResponse
from backend.services.ingestion.ingestion import DocumentIngestionService
from backend.services.ingestion.loader import DocumentLoaderService
from backend.services.ingestion.vector_store import VectorStoreService
from backend.graphs.ingestion.graph import document_graph

router = APIRouter()

def get_ingestion_service():
    return DocumentIngestionService(
        loader=DocumentLoaderService(),
        graph=document_graph,
        vector_store=VectorStoreService()
    )

@router.post("/ingest", response_model=ApiIngestionResponse)
def injest_document(
    file: UploadFile | None = File(None),
    text: str | None = None,
    service: DocumentIngestionService = Depends(get_ingestion_service)):

    return service.ingest_document(file=file, text=text)


@router.post("/ingest/stream")
async def ingest_document_stream(
    file: UploadFile | None = File(None),
    text: str | None = None,
    service: DocumentIngestionService = Depends(get_ingestion_service)):
    """Stream progress updates during document ingestion"""
    
    async def event_generator():
        try:
            # Create event queue for this request
            queue = asyncio.Queue()
            service.set_event_queue(queue)
            
            # Start ingestion task
            task = asyncio.create_task(service.ingest_document_async(file=file, text=text))
            
            # Stream events as they're emitted
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    continue
            
            # Get final result and send it
            try:
                result = await task
                yield f"data: result|{json.dumps(result)}\n\n"
            except Exception as e:
                yield f"data: error|{str(e)}\n\n"
        except Exception as e:
            yield f"data: error|{str(e)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
