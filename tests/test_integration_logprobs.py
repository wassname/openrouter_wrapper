import pytest
import asyncio
from openrouter_wrapper.logprobs import eval_message_choice_logp


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_eval_message_choice_logp():
    """Basic check: evaluate logprobs for choices."""
    model_id = "openai/gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "Answer with a number 1-5."},
        {"role": "user", "content": "Rate this."},
        {"role": "assistant", "content": "\n"},
    ]
    arr, response, logp_dict = await eval_message_choice_logp(messages, model_id)
    assert len(arr) == 5
    assert 'choices' in response
    print(f"Logprobs array length: 5, dict keys: {len(logp_dict)}")
