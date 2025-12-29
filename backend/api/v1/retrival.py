from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from backend.models.retrival import ApiRetrivalRequest, ApiRetrivalResponse
from backend.services.retrival.retrival import RetrivalService
from backend.graphs.retrival.graph import RetrivalGraph
from backend.services.retrival.chunks_retrival import ChunksRetrivalService
from backend.services.retrival.embedding_service import EmbeddingService
from backend.core.embedding_client import get_embedding_client
from backend.core.db.weaviate_client import get_weaviate_client


router = APIRouter()

def get_embedding_service(client = Depends(get_embedding_client)):
    return EmbeddingService(client=client)

def get_chunks_retrival_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    return ChunksRetrivalService(
        weaviate_client=Depends(get_weaviate_client),
        embedding_service=embedding_service
    )

def get_retrival_service():
    retrival_graph = RetrivalGraph()
    return RetrivalService(
        graph=retrival_graph._build_graph(),
        # chunks_retrival_service=get_chunks_retrival_service()
    )

@router.post("/retrival", response_model=ApiRetrivalResponse)
async def retrival_endpoint(
    request: ApiRetrivalRequest,
    retrival_service: RetrivalService = Depends(get_retrival_service)
):
    try:
        response = retrival_service.retrieve(request.query, request.document_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))