from backend.utils.decorators import track_execution_time

class RetrivalService:

    def __init__(self,graph):
        self.graph = graph

    @track_execution_time
    def retrieve(self, query: str, document_id:str) -> dict:
        '''
        Given a query and document_id,retrieve final answers from vector store
        and return the response from the retrival graph
        '''
        graph_output = self.graph.invoke({
            "query": query,
            "document_id": document_id
        })
        
        # Extract final response from the last assistant message
        messages = graph_output.get("messages", [])
        final_response = None
        if messages:
            for msg in reversed(messages):
                if msg.type == "ai" and not getattr(msg, 'tool_calls', None):
                    # Handle content as string or list of content blocks
                    content = msg.content
                    if isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                        final_response = "".join(text_parts)
                    else:
                        final_response = content
                    break
        
        # Extract source chunks
        source_chunks = graph_output.get("retrived_chunks", [])
        
        # Extract web results
        web_results = graph_output.get("web_results", [])
        
        # Build tool calls history
        tool_calls_history = []
        for i, msg in enumerate(messages):
            if msg.type == "ai" and getattr(msg, 'tool_calls', None):
                tool_call = msg.tool_calls[0]  # Assuming one tool per message
                history_entry = {
                    "step_number": i,
                    "tool_name": tool_call.get("name"),
                    "parameters": tool_call.get("args", {}),
                    "query": tool_call.get("args", {}).get("query", ""),
                    # Note: Retrieved chunks per call not directly tracked; using accumulated
                }
                tool_calls_history.append(history_entry)
        
        # Build conversation messages
        conversation_messages = []
        for msg in messages:
            # Normalize content
            content = msg.content
            if isinstance(content, list):
                text_parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
                normalized_content = "".join(text_parts)
            else:
                normalized_content = content
            
            msg_dict = {
                "role": msg.type,  # e.g., "human", "ai", "tool"
                "content": normalized_content,
            }
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            conversation_messages.append(msg_dict)
        
        return {
            "status": "success",
            "final_response": final_response,
            "source_chunks": source_chunks,
            "web_results": web_results,
            "tool_calls_history": tool_calls_history,
            "conversation_messages": conversation_messages,
            "error_message": None
        }