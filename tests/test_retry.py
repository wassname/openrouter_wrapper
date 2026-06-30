import httpx

from openrouter_wrapper.retry import ProviderError, RetryException, is_retryable_error


def _http_error(status_code: int, text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=text)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_retryable_http_status_codes_are_checked() -> None:
    assert is_retryable_error(_http_error(429), "rate limited")
    assert is_retryable_error(_http_error(503), "upstream unavailable")
    assert not is_retryable_error(_http_error(403), "moderation")


def test_free_model_daily_limit_is_not_retryable() -> None:
    assert not is_retryable_error(_http_error(429), "free-models-per-day")


def test_retryable_provider_error_patterns() -> None:
    exc = ProviderError(
        "Provider returned error",
        data={"error": {"message": "qwen/qwen3-8b is temporarily rate-limited upstream"}},
    )
    assert is_retryable_error(exc)


def test_retry_exception_is_retryable() -> None:
    assert is_retryable_error(RetryException("retry me"))
