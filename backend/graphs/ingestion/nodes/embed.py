from openai import AzureOpenAI
from backend.graphs.ingestion.state import RagIngestState
from backend.core.config import config


def get_embedding_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=config.azure_api_version,
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key
    )


def generate_embeddings(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    client = get_embedding_client()
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=config.text_embedding_model
        )
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def embedd(state: RagIngestState) -> RagIngestState:
    chunks = state["chunks"]
    
    # Use contextualized chunk if available, otherwise use content
    texts = [
        chunk.get("contextualized_chunk") or chunk.get("content", "")
        for chunk in chunks
    ]
    
    embeddings = generate_embeddings(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return state