from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

class RetrieveTool:
    def __init__(self, retrival_methods, reranking_service):
        """
        Initialize with injected retrieval and reranking services.
        """
        self.retrival_methods = retrival_methods
        self.reranking_service = reranking_service

    def get_tool(self):
        """
        Returns a clean tool function for the LLM. 
        This closure captures 'self' so the LLM never sees it as a parameter.
        """

        @tool
        def retrieve_from_knowledge_base(
            query: str,
            vector_search_top_k: int,
            bm25_top_k: int,
            reranking_top_k: int,
            state: Annotated[dict, InjectedState],
        ) -> List[Dict[str, Any]]:
            """
            PRIMARY RETRIEVAL TOOL: Search the internal knowledge base to answer user queries.

            This tool MUST be called first for every query. It uses a hybrid search approach:
            1. Vector semantic search – finds conceptually similar content
            2. BM25 keyword search – finds exact term matches
            3. Reciprocal Rank Fusion – combines both results
            4. Cross-encoder reranking – ranks results by relevance to the query

            Use this tool repeatedly with different queries if initial results are insufficient.
            After gathering sufficient information from retrieval, use websearch_tool for validation.

            Args:
                query (str): The search query for the internal knowledge base.
                vector_search_top_k (int): Number of vector search results (typically 5–10).
                bm25_top_k (int): Number of BM25 keyword search results (typically 5–10).
                reranking_top_k (int): Final results to return after reranking (typically 3–5).
                state: Graph state containing document_id and retrieval history.

            Returns:
                List[dict]: Retrieved chunks with breadcrumb, contextualized text,
                page/line metadata, and reranking scores.
            """
            print("Retrieve Tool Invoked with query:", query)
            # --- Retrieval Logic ---
            # We use 'self' here because it's available in the outer scope
            vector_search_results = self.retrival_methods.vector_search(
                query=query,
                document_id=state.get("document_id"),
                top_k=vector_search_top_k,
            )

            bm25_results = self.retrival_methods.bm25_search(
                query=query,
                document_id=state.get("document_id"),
                top_k=bm25_top_k,
            )

            # --- Fusion & Reranking ---
            results = self.retrival_methods.reciprocal_rank_fusion(
                vector_results=vector_search_results,
                bm25_results=bm25_results
            )

            reranked_results = self.reranking_service.rerank(
                query=query,
                chunks=results,
                top_k=reranking_top_k
            )

            # --- Formatting Output ---
            filtered_results = [
                {
                    "breadcrumbs": chunk.get("breadcrumbs", ""),
                    "contextualized_chunk": chunk.get("contextualized_chunk", ""),
                    "line_number": chunk.get("line_number"),
                    "page_number": chunk.get("page_number"),
                    "reranking_score": chunk.get("reranking_score")
                }
                for chunk in reranked_results
            ]

            # Update state with full results for background history
            # Note: InjectedState allows you to read/mutate the state dict directly
            state.setdefault("retrieved_chunks", []).extend(reranked_results)
            # print("Retrieve Tool Results:", filtered_results)
            return filtered_results

        return retrieve_from_knowledge_base