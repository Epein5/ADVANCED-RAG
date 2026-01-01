import atexit
import redis
from typing import Optional
from backend.core.config import config


class RedisClientManager:
    """
    Singleton Redis client manager with health check on connection.
    Ensures Redis is available before returning client instance.
    """
    
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get or create Redis client instance."""
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance
    
    @classmethod
    def _create_client(cls) -> redis.Redis:
        """Create Redis client with health check."""
        try:
            client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Health check
            client.ping()
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    @classmethod
    def setup(cls) -> None:
        """
        Initialize Redis connection with health check.
        Call this during app startup to ensure Redis is available.
        Raises ConnectionError if Redis is unavailable.
        """
        try:
            cls.get_client()  # Already pings during creation
            print("✓ Redis connection verified")
        except ConnectionError as e:
            print(f"✗ Redis connection failed: {str(e)}")
            raise
    
    @classmethod
    def close_client(cls) -> None:
        """Close Redis connection."""
        if cls._instance is not None:
            try:
                cls._instance.close()
            finally:
                cls._instance = None


def get_redis_client() -> redis.Redis:
    """Dependency injection function for FastAPI."""
    return RedisClientManager.get_client()


# Auto-cleanup on app exit
atexit.register(RedisClientManager.close_client)
