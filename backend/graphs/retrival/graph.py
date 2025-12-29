from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.graphs.retrival.state import RetrivalGraphState
from backend.graphs.retrival.nodes.orchestrator import orchestrator_node
from backend.graphs.retrival.tools.web_search import websearch_tool
from backend.graphs.retrival.tools.retrival import retrieve_tool
from backend.graphs.retrival.tools.summarizer import summarizer_tool
from backend.core.llm_client import get_llm


class RetrivalGraph:
    def __init__(self) -> None:
        self.llm = get_llm().bind_tools([retrieve_tool, websearch_tool, summarizer_tool])
        self.tools = [
            websearch_tool,
            retrieve_tool,
            summarizer_tool,
        ]
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(RetrivalGraphState)
        
        # Inject LLM into orchestrator using partial (Dependency Injection)
        orchestrator_with_llm = partial(orchestrator_node, llm=self.llm)
        
        graph.add_node("orchestrator", orchestrator_with_llm)
        graph.add_node("tools", ToolNode(self.tools))
        
        graph.set_entry_point("orchestrator")
        
        def should_continue(state: RetrivalGraphState):
            if "tool_calls" in state and state["tool_calls"]:
                return "tools"
            else:
                return "end"
        
        graph.add_conditional_edges(
            "orchestrator",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        graph.add_edge("tools", "orchestrator")
        
        return graph.compile()