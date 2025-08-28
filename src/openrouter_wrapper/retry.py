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


RETRYABLE_STATUS_CODES=[
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

def is_retryable_error(exc: Exception, response_text: str = "") -> bool:
    """
    Determine if an error is retryable based on type, status code, and message patterns.
    Logs all errors for debugging.
    """
    logger.info(f"Evaluating error for retry: {exc}")
    
    if isinstance(exc, RetryException):
        logger.warning(f"Retryable: Explicit RetryException - {exc}")
        return True
    
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RequestError)):
        logger.warning(f"Retryable: Network/timeout error - {exc}")
        return True
    
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        logger.error(f"HTTP error {status_code}: {exc.response.text}")
        
        if status_code in RETRYABLE_STATUS_CODES:
            return True    
        
        # Check for specific retryable patterns in response (e.g., upstream rate limits)
        retry_patterns = [
            "rate-limited upstream",
            "temporarily rate-limited upstream",
            "provider returned error",
            "please try again"
        ]
        for pattern in retry_patterns:
            if pattern in response_text.lower():
                logger.warning(f"Retryable: Pattern '{pattern}' in response - {response_text}")
                return True
        
        logger.error(f"Non-retryable: Status {status_code} - {exc}. response_text:{response_text}")
        return False
    
    # Log unexpected errors but don't retry by default
    logger.error(f"Non-retryable: Unexpected error - {exc}")
    return False


@stamina.retry(on=is_retryable_error, attempts=5, wait_max=20)
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
    # TODO go through this for ways to improve retry vs error raising https://old.reddit.com/r/JanitorAI_Official/comments/1m7r5ti/openrouter_error_guide_so_you_dont_have_to_scroll/
    # https://openrouter.ai/docs/api-reference/errors#error-codes
    # 403 is moderation
    # 429 is request rate limit exceeded
    try:
        response.raise_for_status()
    except (requests.exceptions.HTTPError, httpx.HTTPStatusError) as e:
        # requests.exceptions.HTTPError: 408 Client Error: Request Timeout for url: https://openrouter.ai/api/v1/chat/completions
        logger.error(f"HTTP error for model {model_id}: {response.text}")
        if not is_retryable_error(e, response.text):
            raise  # Non-retryable, stop here
        raise RetryException(f"Retryable HTTP error: {e}")  # Trigger retry
    try:
        data = response.json()
        
        if "error" in data:
            error_msg = data['error'].get('message', str(data['error']))
            logger.error(f"OpenRouter returned error: {error_msg}")
            raise ProviderError(error_msg, data=data)

        if "choices" not in data:
            raise MalformedResponseError(f"{model_id} response missing 'choices' field", data=data)
        
        if not data['choices']:
            raise MalformedResponseError(f"{model_id} returned empty choices", data=data)

        choice = data['choices'][0]
        if choice['finish_reason'] == 'error':
            error = choice['error']['message']
            logger.error(f"Upstream error for model {model_id}: {error}")

            # e.g. open_router.logprobs.UpstreamError: Upstream error from Cerebras: Encountered a server error
            if 'please try again' in str(error).lower():
                raise RetryException("Upstream server error, retrying")
            raise UpstreamError(error, data=data)

    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response for model {model_id}: {e}")
        logger.debug(f"Response text: {response.text}")
        raise MalformedResponseError(f"Malformed response for model {model_id}", data=response.text)
    except OpenRouterError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error for model {model_id}: {e}")
        logger.debug(response.text)
        # logger.debug(response.headers)
        raise e
    
    return data
