from tavily import TavilyClient
from backend.core.config import config


def get_tavily_client() -> TavilyClient:
    """
    Get or create a Tavily client instance for web search.
    Follows the same pattern as other client initializations.
    """
    return TavilyClient(api_key=config.tavily_api_key)


# Create a singleton instance
tavily_client = get_tavily_client()
