from fastapi import APIRouter, Depends, HTTPException
from backend.models.retrival import ApiRetrivalRequest, ApiRetrivalResponse
from backend.services.retrival.retrival import RetrivalService
from backend.services.retrival.conversation import ConversationService
from backend.graphs.retrival.graph import RetrivalGraph
from backend.services.retrival.chunks_retrival import ChunksRetrivalService
from backend.services.retrival.embedding_service import EmbeddingService
from backend.services.retrival.reranking import RerankingService
from backend.core.embedding_client import get_embedding_client
from backend.core.db.weaviate_client import get_weaviate_client
import traceback


router = APIRouter()

def get_embedding_service(client = Depends(get_embedding_client)):
    return EmbeddingService(embedding_client=client)

def get_chunks_retrival_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    weaviate_client = Depends(get_weaviate_client)
):
    return ChunksRetrivalService(
        weaviate_client=weaviate_client,
        embedding_service=embedding_service
    )

def get_reranking_service():
    return RerankingService()

def get_conversation_service():
    return ConversationService()

def get_retrival_service(
    chunks_retrival_service: ChunksRetrivalService = Depends(get_chunks_retrival_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
    conversation_service: ConversationService = Depends(get_conversation_service)
):
    retrival_graph = RetrivalGraph(
        retrival_methods=chunks_retrival_service,
        reranking_service=reranking_service
    )
    return RetrivalService(
        graph=retrival_graph.graph,
        conversation_service=conversation_service
    )

@router.post("/retrival", response_model=ApiRetrivalResponse)
async def retrival_endpoint(
    request: ApiRetrivalRequest,
    retrival_service: RetrivalService = Depends(get_retrival_service)
):
    try:
        response = retrival_service.retrieve_with_conversation(
            request.query,
            request.document_id
        )
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))