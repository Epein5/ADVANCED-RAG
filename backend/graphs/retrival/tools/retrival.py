from backend.graphs.retrival.state import RetrivalGraphState
from langchain_core.tools import tool


@tool
def retrieve_tool(query: str) -> str:
    '''
    Use this tool to retrieve information from your internal knowledge base.

    Args:
        query (str): The user's query to search for.
        top_k (int): The number of top relevant documents to retrieve.
    '''
    pass