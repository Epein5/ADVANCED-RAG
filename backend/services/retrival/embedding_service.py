from openai import AzureOpenAI
from backend.utils.decorators import track_execution_time
from backend.core.config import config

class EmbeddingService:
    '''Generate embeddings for given text inputs using OpenAI.'''
    
    def __init__(self, embedding_client: AzureOpenAI):
        self.embedding_client = embedding_client
    
    @track_execution_time
    def embedd_text(self, text: str) -> list[float]:
        """
        Generate embeddings for multiple texts.
        Returns embeddings as list of vectors.
        """
        response = self.embedding_client.embeddings.create(
            model= config.text_embedding_model,
            input=text
        )
        return response.data[0].embedding