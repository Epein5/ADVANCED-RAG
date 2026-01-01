from backend.graphs.retrival.state import RetrivalGraphState
from backend.core.llm_client import get_llm
from langchain.messages import SystemMessage, HumanMessage


def orchestrator_node(state: RetrivalGraphState, llm) -> RetrivalGraphState:
    # LLM already has tools bound (injected from graph.py)
    # Use llm.invoke() to call the model with tools
    
    # Build system prompt with retrieved context if available
    base_prompt = """You are an AI Research Orchestrator. Follow this protocol strictly for thorough, accurate research.

WORKFLOW:
1. RETRIEVAL: Break the main query into 2-3 specific sub-queries. Call retrieve_from_knowledge_base for EACH sub-query separately to gather comprehensive information. Be thorough - call retrieval multiple times with different angles (e.g., definitions, requirements, exceptions, examples).
2. ANALYSIS: After gathering sufficient document data, analyze what you have. If information is incomplete, call retrieval again with more specific queries.
3. WEB SEARCH: Call websearch_tool ONCE for external validation and recent updates.
4. FINAL ANSWER: Synthesize all findings with proper citations.

RULES:
- Generate meaningful, specific sub-queries for retrieval (e.g., "[main topic] definitions and scope", "[main topic] requirements and procedures", "[main topic] exceptions and limitations").
- Call retrieve_from_knowledge_base at least 2-3 times with different queries before considering web search.
- Structure the final answer with clear sections:
  * DOCUMENT FINDINGS: Summarize info from retrieved document chunks, citing specific breadcrumbs, page numbers, and line numbers (e.g., "According to 'Chapter 1 > Section 2' on page 5, line 10...").
  * WEB SEARCH VALIDATION: Summarize web search results in a separate paragraph/section.
  * FINAL CONCLUSION: Combine both sources for the complete answer.
- Always cite sources explicitly with location details for credibility.

TASK: {query}
"""
    
    retrieved_chunks = state.get("retrived_chunks", [])
    if retrieved_chunks:
        # Add retrieved context to prompt
        context_str = "\n\nRETRIEVED CONTEXT:\n" + "\n".join([
            f"- Breadcrumbs: {chunk.get('breadcrumbs', 'N/A')}\n  Page: {chunk.get('page_number', 'N/A')}, Line: {chunk.get('line_number', 'N/A')}\n  Content: {chunk.get('contextualized_chunk', chunk.get('content', ''))[:500]}..."
            for chunk in retrieved_chunks[:10]  # Limit to top 10 for prompt length
        ])
        base_prompt += context_str + "\n\nUse this context to provide detailed, cited answers."
    
    system_prompt = SystemMessage(content=base_prompt.format(query=state.get('query', '')))
    
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
