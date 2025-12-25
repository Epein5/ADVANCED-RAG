from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

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
