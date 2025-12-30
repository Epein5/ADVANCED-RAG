from backend.graphs.retrival.state import RetrivalGraphState
from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedState


@tool
def retrieve_tool(query: str, vector_search_top_k: int, bm25_top_k: int, reranking_top_k: int, state: Annotated[dict, InjectedState], retrival_methods, reranking_service) -> str:
    '''
    PRIMARY RETRIEVAL TOOL: Search the internal knowledge base to answer user queries.
    
    This tool MUST be called first for every query. It uses a hybrid search approach:
    1. Vector semantic search - finds conceptually similar content
    2. BM25 keyword search - finds exact term matches
    3. Reciprocal rank fusion - combines both results
    4. Cross-encoder reranking - ranks by relevance to the query
    
    Use this tool repeatedly with different queries if initial results are insufficient.
    After gathering sufficient information from retrieval, use websearch_tool for validation.

    Args:
        query (str): The search query for the internal knowledge base.
        vector_search_top_k (int): Number of vector search results (typically 5-10).
        bm25_top_k (int): Number of BM25 keyword search results (typically 5-10).
        reranking_top_k (int): Final results to return after reranking (typically 3-5).
        state: Graph state containing document_id and message history.
        retrival_methods: Service for performing hybrid searches.
        reranking_service: Service for cross-encoder reranking.
    
    Returns:
        List of retrieved chunks with relevance scores, breadcrumbs, and contextual information.
    '''
    
    vector_search_results = retrival_methods.vector_search(
        query=query,
        document_id=state["document_id"],
        top_k=vector_search_top_k,
    )
    bm25_results = retrival_methods.bm25_search(
        query=query,
        document_id=state["document_id"],
        top_k=bm25_top_k,
    )
    results = retrival_methods.reciprocal_rank_fusion(
        vector_results=vector_search_results,
        bm25_results=bm25_results
    )
    
    # Re-rank results using cross-encoder
    reranked_results = reranking_service.rerank(
        query=query,
        chunks=results,
        top_k=reranking_top_k
    )
    
    # Filter results to only include necessary fields
    filtered_results = []
    for chunk in reranked_results:
        filtered_chunk = {
            "breadcrumbs": chunk.get("breadcrumbs", []),
            "contextualized_chunk": chunk.get("contextualized_chunk", ""),
            "line_number": chunk.get("line_number"),
            "page_number": chunk.get("page_number"),
            "reranking_score": chunk.get("reranking_score")
        }
        filtered_results.append(filtered_chunk)
    
    # Save full original chunks to state
    if "retrieved_chunks" not in state:
        state["retrieved_chunks"] = []
    state["retrieved_chunks"].extend(reranked_results)
    
    return filtered_results