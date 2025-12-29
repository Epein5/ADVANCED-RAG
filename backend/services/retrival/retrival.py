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
        graph_output = {"status":"1"}
        return graph_output