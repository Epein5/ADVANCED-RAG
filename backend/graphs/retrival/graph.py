from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.graphs.retrival.state import RetrivalGraphState
from backend.graphs.retrival.nodes.orchestrator import orchestrator_node
from backend.graphs.retrival.tools.web_search import websearch_tool
from backend.graphs.retrival.tools.retrival import RetrieveTool
from backend.graphs.retrival.tools.summarizer import summarizer_tool
from backend.core.llm_client import get_llm



class RetrivalGraph:
    def __init__(self, retrival_methods, reranking_service) -> None:
        self.retrival_methods = retrival_methods
        self.reranking_service = reranking_service
        self.graph = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(RetrivalGraphState)
        
        # Create partial tool with retrival_methods and reranking_service injected
        retrieve_tool_instance = RetrieveTool(
            retrival_methods=self.retrival_methods,
            reranking_service=self.reranking_service
        )
        
        # Create tools list with partial version
        tools = [
            websearch_tool,
            retrieve_tool_instance.get_tool(),
        ]
        
        # Bind tools to LLM (with partial versions)
        llm = get_llm().bind_tools(tools)
        
        # Inject LLM into orchestrator using partial (Dependency Injection)
        orchestrator_with_llm = partial(orchestrator_node, llm=llm)

        graph.add_node("orchestrator", orchestrator_with_llm)

        graph.add_node("tools", self._tool_node_with_state_tracking(ToolNode(tools)))
        
        graph.set_entry_point("orchestrator")
        
        def should_continue(state: RetrivalGraphState):
            # Stop if no tool calls or if both steps are complete
            if not state.get("tool_calls"):
                return "end"
            if state.get("retrieval_completed") and state.get("websearch_completed"):
                return "end"
            return "tools"
        
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
    
    def _tool_node_with_state_tracking(self, tool_node):
        """Wrap ToolNode to track completion flags after tool execution"""
        def wrapped_tool_node(state: RetrivalGraphState):
            # Execute tools
            result = tool_node.invoke(state)
            
            # Track which tools were called
            if state.get("tool_calls"):
                for tool_call in state["tool_calls"]:
                    if tool_call.get("name") == "retrieve_from_knowledge_base":
                        # Check if we got results and store them
                        result["retrieval_completed"] = True
                        # Store the retrieved content for context
                        if "content" in result:
                            result["retrieval_results"] = str(result.get("content", ""))
                    elif tool_call.get("name") == "web_search":
                        result["websearch_completed"] = True
            
            return result
        
        return wrapped_tool_node