from backend.graphs.ingestion.state import RagIngestState
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from backend.core.config import config
import asyncio
import warnings

from backend.utils.decorators import track_execution_time

# Suppress aiohttp's unclosed transport warnings (known issue with google-genai async client)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport")

class ChunkResponse(BaseModel):
    context: str = Field(description="Concise context explaining where this chunk sits within the document.")
    chunk_type: str = Field(description="Classification or type of the chunk (e.g., 'section', 'definition').")
    breadcrumb: str = Field(description="Hierarchical path or breadcrumb showing chunk location in document structure.")


def _build_chunk_prompt(chunk_content: str) -> str:
    """Build the prompt for contextual retrieval."""
    return f"""
    Here is a specific chunk from the document provided in the context:
    <chunk>
    {chunk_content}
    </chunk>

    Generate retrieval context for this chunk. The context should:
    - Be 2-3 sentences that situate this chunk within the document
    - Include key terms/topics from surrounding sections
    - Explain the chunk's purpose or what question it answers
    - Be self-contained (assume the reader only sees the chunk + your context)
    """


def _calculate_backoff(attempt: int) -> int:
    """Calculate exponential backoff time, capped at 60 seconds."""
    return min(5 * (2 ** (attempt - 1)), 60)


def _update_chunk_with_result(chunk: dict, result: ChunkResponse) -> None:
    """Update a chunk dict with the contextual retrieval result."""
    chunk['context'] = result.context
    chunk['chunk_type'] = result.chunk_type
    chunk['breadcrumbs'] = [result.breadcrumb]
    chunk['contextualized_chunk'] = f"{result.context}\n\n{chunk['content']}"


def _delete_cache_safely(client: genai.Client, cache_name: str) -> None:
    """Safely delete a cache, logging any errors."""
    try:
        client.caches.delete(name=cache_name)
    except Exception as e:
        print(f"Error deleting cache: {e}")

@track_execution_time
def contextual_retrival_node(state: RagIngestState) -> RagIngestState:
    """
    Applies Contextual Retrieval techniques to enhance chunk relevance and context.
    Also gathers metadata.
    """
    client = genai.Client(api_key=config.google_api_key)
    async_client = genai.Client(api_key=config.google_api_key, http_options=types.HttpOptions(api_version="v1beta"))
    print("Starting contextual retrieval node...")
    
    document_content = "\n\n".join([c['content'] for c in state['chunks']])
    
    cache = client.caches.create(
        model=config.llm_model,
        config=types.CreateCachedContentConfig(
            display_name='ingestion_cache',
            system_instruction='You are a helpful assistant specialized in document analysis.',
            contents=[document_content],
            ttl='600s'
        )
    )
    
    try:
        async def process_chunk(semaphore: asyncio.Semaphore, chunk: dict, index: int) -> ChunkResponse:
            async with semaphore:
                prompt = _build_chunk_prompt(chunk['content'])
                attempt = 0
                while True:
                    try:
                        response = await async_client.aio.models.generate_content(
                            model=config.llm_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                cached_content=cache.name,
                                response_mime_type='application/json',
                                response_schema=ChunkResponse
                            )
                        )
                        return response.parsed
                    except Exception as e:
                        attempt += 1
                        wait_time = _calculate_backoff(attempt)
                        print(f"Error on chunk {index}: {e}, attempt {attempt}, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)

        async def process_all_chunks() -> list:
            semaphore = asyncio.Semaphore(10)
            tasks = [process_chunk(semaphore, chunk, i) for i, chunk in enumerate(state['chunks'])]
            results = await asyncio.gather(*tasks)
            # Allow pending async tasks to complete and connections to close
            await asyncio.sleep(0.1)
            return results

        results = asyncio.run(process_all_chunks())
        
        for i, result in enumerate(results):
            if result:
                _update_chunk_with_result(state['chunks'][i], result)

    finally:
        _delete_cache_safely(client, cache.name)
    
    return state