import logging
import time
from functools import wraps
from typing import Any, Callable
import asyncio

logger = logging.getLogger(__name__)


def track_execution_time(func: Callable) -> Callable:
    """
    Decorator to track and log the execution time of a function.
    Works with both sync and async functions.
    """
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = func.__name__
        
        try:
            logger.info(f"Starting execution: {func_name}")
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(f"Completed {func_name} in {execution_time:.4f} seconds")
    
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = func.__name__
        
        try:
            logger.info(f"Starting execution: {func_name}")
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(f"Completed {func_name} in {execution_time:.4f} seconds")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
