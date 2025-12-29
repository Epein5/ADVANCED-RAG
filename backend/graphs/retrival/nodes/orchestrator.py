from backend.graphs.retrival.state import RetrivalGraphState
from backend.core.llm_client import get_llm
from langchain.messages import SystemMessage


def orchestrator_node(state: RetrivalGraphState, llm) -> RetrivalGraphState:
    # LLM already has tools bound (injected from graph.py)
    # Use llm.invoke() to call the model with tools
    # print("Orchestrator Node Invoked with state:", state)
    system_prompt = SystemMessage(
    content = """
        You are an AI Orchestrator with a strict multi-step verification protocol.

        ### STEP 1: RETRIEVAL (MANDATORY)
        Whenever a user asks a question, your FIRST action must ALWAYS be to use the 'retrieval_tool'. You must do this even if you think you know the answer. 

        ### STEP 2: EVALUATION
        Once you receive the results from the retrieval_tool:
        - If the information is missing or unclear, use the 'retrieval_tool' again with a different query.
        - If the information is sufficient, proceed to Step 3.

        ### STEP 3: WEB SEARCH (VALIDATION ONLY)
        After you have your answer from Retrieval, you must call the 'web_search' tool to VALIDATE the facts. 
        - Use the web search to check for any contradictions or very recent updates.
        - Do NOT use web search as your primary source of information.

        ### STEP 4: FINAL ANSWER
        Only after you have both Retrieval results and Web Search validation should you provide a final response to the user. Summarize the findings and mention if the web search confirmed the internal data.
    """)
    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages":[response]}
