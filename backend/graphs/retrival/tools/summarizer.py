from backend.graphs.retrival.state import RetrivalGraphState
from langchain_core.tools import tool


@tool
def summarizer_tool(query: str) -> str:
    '''
    To be implemented: Tool to perform retrieval based on the query
    '''
    pass