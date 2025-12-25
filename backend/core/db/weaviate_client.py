import atexit
import weaviate
from typing import Optional
from backend.core.config import config


class WeaviateClientManager:
    
    _instance: Optional[weaviate.WeaviateClient] = None
    
    @classmethod
    def get_client(cls) -> weaviate.WeaviateClient:
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance
    
    @classmethod
    def _create_client(cls) -> weaviate.WeaviateClient:
        try:
            client = weaviate.connect_to_local(
                host=config.weaviate_host,
                port=config.weaviate_port,
            )
            client.is_ready()
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {str(e)}")
    
    @classmethod
    def close_client(cls) -> None:
        if cls._instance is not None:
            try:
                cls._instance.close()
            finally:
                cls._instance = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    return WeaviateClientManager.get_client()


# Auto-cleanup on app exit
atexit.register(WeaviateClientManager.close_client)
