
from typing import Dict, Optional
import httpx
import stamina
from loguru import logger
    
stamina.instrumentation.set_on_retry_hooks([stamina.instrumentation.LoggingOnRetryHook])

class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors"""
    def __init__(self, message, data: Optional[Dict] = None):
        super().__init__(message)
        self.data = data

class LogprobsNotSupportedError(OpenRouterError):
    """Model doesn't support logprobs"""
    pass

class ProviderError(OpenRouterError):
    """Provider-specific error"""
    pass

class MalformedResponseError(OpenRouterError):
    """Response structure is not as expected"""
    pass

class LowProbabilityError(OpenRouterError):
    """Model returned unexpectedly low probabilities"""
    pass


class UpstreamError(OpenRouterError):
    """An error occurred in the upstream model response"""


status_forcelist=[
    # 403 # is moderation
    408,  # request timeout
    429, # request rate limit exceeded
    500,
    502, # 
    503,
    504
],  # the HTTP status codes to retry on


class RetryException(Exception):
    """Exception raised for retryable errors."""
    pass

def retry_only_on_real_errors(exc: Exception) -> bool:
    logger.info(f"retry_only_on_real_errors called with {exc}")
    if isinstance(exc, RetryException):
        return True
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RequestError)):
        logger.info(f"Retrying due to httpx network error: {exc}")
        return True
    # If the error is an HTTP status error, only retry on 5xx errors.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in status_forcelist
    # Otherwise retry on all httpx errors.
    logger.warning(f"Got unexpected error e={exc}")
    return False
    # return isinstance(exc, httpx.HTTPError)

    # httpcore.ReadTimeout: The read operation timed out
    # httpx.HTTPStatusError: Server error '502 Bad Gateway' for url 'https://openrouter.ai/api/v1/chat/completions'
