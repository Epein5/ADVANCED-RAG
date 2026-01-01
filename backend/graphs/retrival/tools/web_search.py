from backend.graphs.retrival.state import RetrivalGraphState
from backend.core.tavily_client import tavily_client
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from backend.utils.decorators import track_execution_time

@tool
@track_execution_time
def websearch_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    topic: str = "general",
    state: Annotated[dict, InjectedState] = {}
) -> str:
    '''
    VALIDATION TOOL: Perform web search to validate and cross-check retrieval results.
    
    This tool should ONLY be called AFTER internal retrieval has been completed and 
    sufficient information has been gathered. Use it to:
    - Verify facts from internal knowledge base against current web sources
    - Check for recent updates or contradictions
    - Validate claims with external sources
    
    Do NOT use this as your primary information source. Always rely on retrieve_tool first.

    Args:
        query (str): The search query for web validation.
        max_results (int): Maximum number of results to return (default: 5).
        search_depth (str): Search depth - "basic" for fast results or "advanced" for comprehensive search (default: "advanced").
        topic (str): Search topic context - "general" for all topics or "news" for recent news (default: "general").
        state: Graph state to update with web results.
    
    Returns:
        JSON with search results, summaries, and AI-generated answer from web sources.
    '''
    
    results = tavily_client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=True,
        topic=topic
    )
    
    # Update state with web results for tracking
    if state:
        state.setdefault("web_results", []).append({
            "query": query,
            "results": results
        })
    
    return results