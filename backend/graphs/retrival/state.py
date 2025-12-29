from typing import TypedDict


class RetrivalGraphState(TypedDict):
    query: str
    document_id: str
    loop_count: int
    retrived_chunks: list[dict]
    final_response: str