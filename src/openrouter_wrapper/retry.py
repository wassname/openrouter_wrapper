
from typing import Dict, Optional
import httpx
import stamina
from loguru import logger
import requests
from . import OPENROUTER_API_KEY

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


@stamina.retry(on=retry_only_on_real_errors, attempts=5, wait_max=20)
def openrouter_request(payload):
    """"Run a completion with logprobs on OpenRouter."""
    model_id = payload.get("model_id")
    response = httpx.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60.0,
    )
    # https://openrouter.ai/docs/api-reference/errors#error-codes
    # 403 is moderation
    # 429 is request rate limit exceeded
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        # requests.exceptions.HTTPError: 408 Client Error: Request Timeout for url: https://openrouter.ai/api/v1/chat/completions
        logger.error(f"Error response: {response.text}")
        raise
    try:
        data = response.json()
        
        if "error" in data:
            raise ProviderError(data['error'], data=data)

        if "choices" not in data:
            raise MalformedResponseError(f"{model_id} response missing 'choices' field", data=data)
        
        if not data['choices']:
            raise MalformedResponseError(f"{model_id} returned empty choices", data=data)

        if data['choices'][0]['finish_reason'] == 'error':
            error = data['choices'][0]['error']['message']
            
            # open_router.logprobs.UpstreamError: Upstream error from Cerebras: Encountered a server error
            if 'please try again' in str(error).lower():
                # raise http 5?? error
                raise RetryException("Server error, please try again later")
            raise UpstreamError(error, data=data)

        if not data['choices'][0].get('logprobs'):
            raise LogprobsNotSupportedError(f"{model_id} has no logprobs capability", data=data)
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response for model {model_id}: {e}")
        logger.debug(f"Response text: {response.text}")
        raise MalformedResponseError(f"Malformed response for model {model_id}", data=response.text)
    except OpenRouterError:
        raise
    except Exception as e:
        # logger.error(f"request {response.request.body}")
        logger.error(f"failed with {model_id},{e}")
        logger.debug(response.text)
        # logger.debug(response.headers)
        raise e
    
    return data
