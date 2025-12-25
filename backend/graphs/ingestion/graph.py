from langgraph.graph import StateGraph
from backend.graphs.ingestion.state import ChunkData, RagIngestState
from backend.graphs.ingestion.nodes.chunk_node import chunk_node
from backend.graphs.ingestion.nodes.contextual_retrival import contextual_retrival_node
from backend.graphs.ingestion.nodes.embed import embedd

graph = StateGraph(RagIngestState)

graph.add_node("chunk_node",chunk_node)
graph.set_entry_point("chunk_node")

graph.add_node("contextual_retrival_node",contextual_retrival_node)
graph.add_edge("chunk_node", "contextual_retrival_node")

graph.add_node("embedd",embedd)
graph.add_edge("contextual_retrival_node", "embedd")

graph.set_finish_point("embedd")
document_graph = graph.compile()