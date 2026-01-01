from dotenv import load_dotenv
from pydantic_settings import BaseSettings,SettingsConfigDict

load_dotenv()


class Config(BaseSettings):

    app_name: str = "Advanced-RAG-Backend"
    google_api_key: str
    azure_openai_api_key: str
    tavily_api_key: str

    azure_openai_endpoint: str = "https://grow-me82mm7z-eastus2.services.ai.azure.com"
    azure_api_version: str = "2024-12-01-preview"
    text_embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gemini-2.5-flash"

    # Chunking configuration
    min_chunk_size: int = 500
    max_chunk_size: int = 2000

    # Weaviate configuration
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_collection_name: str = "Chunk"

    # Contextual retrieval configuration
    contextual_retrieval_concurrency: int = 50

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Conversation history configuration
    conversation_ttl_seconds: int = 604800  # 7 days
    max_conversation_history: int = 20      # Load last 20 messages

    # @property
    # def db_url(self):
    #     return f"sqlite:///./{self.db_name}"

    model_config = SettingsConfigDict(env_file=".env")
config = Config()