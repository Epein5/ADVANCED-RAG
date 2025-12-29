from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
from backend.core.config import config


class LLMClientManager:
    
    _instance: Optional[ChatGoogleGenerativeAI] = None
    
    @classmethod
    def get_client(cls) -> ChatGoogleGenerativeAI:
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance
    
    @classmethod
    def _create_client(cls) -> ChatGoogleGenerativeAI:
        try:
            return ChatGoogleGenerativeAI(
                model=config.llm_model,
                api_key=config.google_api_key,
                temperature=0.7,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize LLM client: {str(e)}")
    
    @classmethod
    def close_client(cls) -> None:
        if cls._instance is not None:
            cls._instance = None


def get_llm() -> ChatGoogleGenerativeAI:
    return LLMClientManager.get_client()
