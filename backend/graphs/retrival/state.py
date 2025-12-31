from typing import TypedDict,Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RetrivalGraphState(TypedDict):
    query: str
    document_id: str
    loop_count: int
    retrived_chunks: list[dict]
    final_response: str
    messages: Annotated[Sequence[BaseMessage],add_messages]
    tool_calls: list[dict]
    retrieval_completed: bool
    websearch_completed: bool
    retrieval_results: str  # Store retrieved content for context