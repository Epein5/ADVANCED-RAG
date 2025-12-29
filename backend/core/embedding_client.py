from openai import AzureOpenAI
from typing import Optional
from backend.core.config import config


class EmbeddingClientManager:
    
    _instance: Optional[AzureOpenAI] = None
    
    @classmethod
    def get_client(cls) -> AzureOpenAI:
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance
    
    @classmethod
    def _create_client(cls) -> AzureOpenAI:
        try:
            return AzureOpenAI(
                api_key=config.azure_openai_api_key,
                api_version=config.azure_api_version,
                azure_endpoint=config.azure_openai_endpoint
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {str(e)}")
    
    @classmethod
    def close_client(cls) -> None:
        if cls._instance is not None:
            cls._instance = None


def get_embedding_client() -> AzureOpenAI:
    return EmbeddingClientManager.get_client()