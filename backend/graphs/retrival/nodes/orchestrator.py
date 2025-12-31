from backend.graphs.retrival.state import RetrivalGraphState
from backend.core.llm_client import get_llm
from langchain.messages import SystemMessage, HumanMessage


def orchestrator_node(state: RetrivalGraphState, llm) -> RetrivalGraphState:
    # LLM already has tools bound (injected from graph.py)
    # Use llm.invoke() to call the model with tools
    
    system_prompt = SystemMessage(
        content=f"""You are an AI Orchestrator. Follow this protocol strictly.

WORKFLOW:
1. RETRIEVAL: Break query into sub-parts (e.g., pros/cons). Call retrieve_from_knowledge_base for each (different queries).
2. WEB SEARCH: Call once for validation after retrieval.
3. FINAL ANSWER: Summarize all findings when done.

RULES:
- Call tools as needed for reasoning.
- Provide final answer directly when complete (no tool calls).

TASK: {state.get('query', '')}
""")
    
    # Build message history
    messages = [system_prompt, HumanMessage(content=state.get("query", ""))]
    
    # Add previous messages for context
    if state.get("messages"):
        messages.extend(state["messages"])
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else [],
    }
