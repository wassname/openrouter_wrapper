import httpx
import pytest

from openrouter_wrapper.retry import (
    MalformedResponseError,
    ProviderError,
    RetryException,
    UpstreamError,
    is_retryable_error,
)


def _http_error(status_code: int, text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request, text=text)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_retryable_status_codes_retry(status_code: int):
    assert is_retryable_error(_http_error(status_code))


def test_non_retryable_status_codes_do_not_retry():
    assert not is_retryable_error(_http_error(400))
    assert not is_retryable_error(_http_error(403), "moderation")


def test_free_model_daily_limit_does_not_retry():
    assert not is_retryable_error(
        _http_error(429),
        response_text="free-models-per-day limit exceeded",
    )


def test_retryable_provider_error_patterns() -> None:
    exc = ProviderError(
        "Provider returned error",
        data={"error": {"message": "qwen/qwen3-8b is temporarily rate-limited upstream"}},
    )
    assert is_retryable_error(exc)


def test_retry_exception_is_retryable() -> None:
    assert is_retryable_error(RetryException("retry me"))


def test_default_provider_preferences_prefer_deepinfra_with_fallback():
    from openrouter_wrapper.logprobs import build_provider_preferences

    assert build_provider_preferences() == {
        "require_parameters": True,
        "allow_fallbacks": True,
        "order": ["deepinfra"],
    }


def test_provider_whitelist_is_hard_restriction_not_deepinfra_preference():
    from openrouter_wrapper.logprobs import build_provider_preferences

    assert build_provider_preferences(provider_whitelist=["fireworks"]) == {
        "require_parameters": True,
        "allow_fallbacks": True,
        "only": ["fireworks"],
    }


def test_retryable_upstream_sse_json_error() -> None:
    exc = UpstreamError(
        "JSON error injected into SSE stream",
        data={"choices": [{"finish_reason": "error"}]},
    )
    assert is_retryable_error(exc)


def test_retryable_upstream_idle_timeout() -> None:
    exc = UpstreamError(
        "Upstream idle timeout exceeded",
        data={"choices": [{"finish_reason": "error", "error": {"code": 504}}]},
    )
    assert is_retryable_error(exc)


def test_retryable_malformed_response_pattern() -> None:
    exc = MalformedResponseError(
        "Malformed response",
        data="JSON error injected into SSE stream",
    )
    assert is_retryable_error(exc)
