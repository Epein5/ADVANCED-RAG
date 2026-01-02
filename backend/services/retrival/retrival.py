from backend.utils.decorators import track_execution_time
from datetime import datetime

class RetrivalService:

    def __init__(self, graph, conversation_service=None):
        self.graph = graph
        self.conversation_service = conversation_service

    @staticmethod
    def _extract_text_content(content) -> str:
        """Extract text from content (string or list of content blocks)."""
        if isinstance(content, list):
            return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
        return content

    @staticmethod
    def _create_message_dict(role: str, content: str, tool_calls=None) -> dict:
        """Create a normalized message dictionary."""
        msg_dict = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": "message"
        }
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        return msg_dict

    def retrieve_with_conversation(self, query: str, document_id: str) -> dict:
        """
        Retrieve with full conversation flow: load history → retrieve → save history.
        This is the main entry point that handles conversation persistence.
        """
        # Load conversation history from Redis
        conversation_history = self.conversation_service.load_conversation(document_id)
        
        # Retrieve response with history
        response = self.retrieve(query, document_id, conversation_history)
        
        # Save updated conversation to Redis
        if response.get("status") == "success":
            self.conversation_service.save_conversation(
                document_id,
                response.get("conversation_messages", [])
            )
        
        return response

    @track_execution_time
    def retrieve(self, query: str, document_id: str, conversation_history: list = None) -> dict:
        '''
        Given a query and document_id, retrieve final answers from vector store
        and return the response from the retrival graph.
        Includes conversation history if provided (only human + ai messages, no tool messages).
        '''
        # Build input messages: history + current query (exclude tool messages)
        messages = []
        if conversation_history:
            # Only include human and ai messages, skip tool messages
            messages.extend([msg for msg in conversation_history if msg.get("role") in ["human", "ai"]])
        
        messages.append(self._create_message_dict("human", query))
        
        graph_output = self.graph.invoke({
            "query": query,
            "document_id": document_id,
            "messages": messages
        })
        
        # Extract final response from the last assistant message
        graph_messages = graph_output.get("messages", [])
        final_response = None
        if graph_messages:
            for msg in reversed(graph_messages):
                if msg.type == "ai" and not getattr(msg, 'tool_calls', None):
                    final_response = self._extract_text_content(msg.content)
                    break
        
        # Extract source chunks and web results
        source_chunks = graph_output.get("retrived_chunks", [])
        web_results = graph_output.get("web_results", [])
        
        # Build tool calls history
        tool_calls_history = [
            {
                "step_number": i,
                "tool_name": msg.tool_calls[0].get("name"),
                "parameters": msg.tool_calls[0].get("args", {}),
                "query": msg.tool_calls[0].get("args", {}).get("query", ""),
            }
            for i, msg in enumerate(graph_messages)
            if msg.type == "ai" and getattr(msg, 'tool_calls', None)
        ]
        
        # Build conversation messages (only human + ai, skip tool messages)
        conversation_messages = list(messages) if messages else []
        conversation_messages.extend([
            self._create_message_dict(msg.type, self._extract_text_content(msg.content), getattr(msg, 'tool_calls', None))
            for msg in graph_messages
            if msg.type in ["human", "ai"]  # Only save human and ai messages
        ])
        
        return {
            "status": "success",
            "final_response": final_response,
            "source_chunks": source_chunks,
            "web_results": web_results,
            "tool_calls_history": tool_calls_history,
            "conversation_messages": conversation_messages,
            "error_message": None
        }