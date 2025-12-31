from backend.graphs.retrival.state import RetrivalGraphState
from backend.core.llm_client import get_llm
from langchain.messages import SystemMessage, HumanMessage


def orchestrator_node(state: RetrivalGraphState, llm) -> RetrivalGraphState:
    # LLM already has tools bound (injected from graph.py)
    # Use llm.invoke() to call the model with tools
    
    # Build context about what has already been done
    execution_context = ""
    if state.get("retrieval_completed"):
        execution_context += f"\n✓ RETRIEVAL STEP COMPLETED\nRetrieved Information:\n{state.get('retrieval_results', 'No results')}\n"
    
    if state.get("websearch_completed"):
        execution_context += "\n✓ WEB SEARCH STEP COMPLETED (validation done)\n"
    
    system_prompt = SystemMessage(
        content=f"""You are an AI Orchestrator with a strict multi-step verification protocol.

EXECUTION STATUS:
{execution_context if execution_context else "No steps completed yet."}

### CRITICAL STOPPING RULES (MUST FOLLOW):
1. If retrieval_completed=True AND websearch_completed=True → STOP and provide your FINAL ANSWER directly (no tool calls).
2. If retrieval_completed=True AND you have good information → Call web_search ONCE for validation, then provide final answer.
3. If you haven't done retrieval yet → Call retrieval_tool FIRST.
4. NEVER call the same tool twice in a row with identical parameters.
5. When all steps are done, output your response WITHOUT calling any tools.

### STEP-BY-STEP WORKFLOW:
STEP 1: RETRIEVAL (MANDATORY)
- Call retrieve_from_knowledge_base if not done yet
- You may call it 2-3 times with DIFFERENT queries if results are unclear

STEP 2: WEB SEARCH (AFTER RETRIEVAL)
- Once retrieval_completed=True, call web_search for validation
- Use web search to check for contradictions or recent updates

STEP 3: FINAL ANSWER
- After both steps are complete, provide your response directly
- Do NOT call any tools when providing the final answer
- Summarize findings and mention if web search confirmed the data

### CURRENT TASK:
User Query: {state.get('query', '')}
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
