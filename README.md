# OpenRouter Wrapper

A Python library that wraps the OpenRouter API, focusing on retry and error handling, log probability (logprobs) support, and utilities for model filtering and soft structuring of outputs. It handles retries and errors reliably, simplifies logprobs extraction, and supports async operations.

## Installation

Use uv for fast Python package management and installation:

```
uv add --editable .
```

Load environment variables with dotenv. Add `OPENROUTER_API_KEY=your_key` to a `.env` file in your project root.

Add testing dependencies (optional):

```
uv add --dev pytest pytest-asyncio
```

## Testing

Run the 3 simple integration tests to check if the core functions run (skips real API calls if no OPENROUTER_API_KEY):

```
pytest tests/ -v --asyncio-mode=auto
```

## Features

- **Retry and Error Handling**: Automatic retries for network timeouts, rate limits (429), server errors (500+), and upstream issues using the `stamina` library. Custom errors like `LogprobsNotSupportedError`, `ProviderError`, etc.
- **Logprobs Support**: Fetch completions with logprobs, respecting provider limits (e.g., xAI: 8, Fireworks: 5). Extract probabilities for choices or tokens.
- **Model Utilities**: Fetch OpenRouter models and endpoints, filter for logprobs support, pricing, and uptime.
- **Soft Structuring**: Parse semi-structured JSON or regex-based outputs when models can't use strict schemas.
- **Async Support**: All API calls are asynchronous using `httpx.AsyncClient`.

The library handles transient errors reliably and simplifies logprobs extraction for applications like choice evaluation or moral reasoning analysis.

## Usage Examples

### Logprobs (Async Example)

Simplified from `scripts/03_openrouter_topk.ipynb`: Evaluate probabilities for numeric choices (e.g., ratings 1-5).

```python
import asyncio
from openrouter_wrapper.logprobs import eval_message_choice_logp

async def logprobs_example():
    model_id = "meta/llama-3-8b-instruct"  # Logprobs-capable model

    messages = [
        {"role": "system", "content": "You are a rater."},
        {"role": "user", "content": "Rate this response on a scale of 1-5."},
        {"role": "assistant", "content": "<think>\n</think>\n"},  # Prefill for numeric output
    ]
    
    choice_logp_arr, response, logp_dict = await eval_message_choice_logp(
        messages, model_id, max_completion_tokens=2
    )
    
    print("Logprobs for choices 1-5:", choice_logp_arr)
    print("Top tokens:", list(logp_dict.keys())[:5])
    print("Cost:", response.get('usage', {}).get('cost', 'N/A'))

asyncio.run(logprobs_example())
```

### Getting Costs

From model endpoints or response usage.

```python
import asyncio
from openrouter_wrapper.models import get_logp_endpoints

async def costs_example():
    model_id = "openai/gpt-3.5-turbo"
    df_end, data = await get_logp_endpoints(model_id)
    if df_end.height > 0:
        print("Cheapest prompt price per million tokens:", df_end['price_prompt'].min())
    
    # From a response (as above)
    # cost = response['usage']['cost']
    # print("Request cost in USD:", cost * 1000)  # Scaled to 1k tokens approx

asyncio.run(costs_example())
```

### Non-Logprobs Completion (Basic Retry Usage)

Simplified from `scripts/03_judge_openrouter_CoT.py`: Basic completion with retry, parse with soft structuring.

```python
import asyncio
from dotenv import load_dotenv
from openrouter_wrapper.retry import openrouter_request
from openrouter_wrapper.soft_structure import extract_answer_from_json

load_dotenv()  # Loads OPENROUTER_API_KEY from .env

async def non_logprobs_example():
    messages = [
        {"role": "system", "content": "Respond in JSON format: {'rating': number between 1-10}"},
        {"role": "user", "content": "How would you rate this idea?"},
    ]
    
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.5,
    }
    
    try:
        response = await openrouter_request(payload)
        content = response['choices'][0]['message']['content']
        
        # Soft parse for JSON-like output
        rating = extract_answer_from_json(content, ['rating'])
        print("Parsed rating:", rating)
        print("Full response:", content)
        print("Tokens used:", response['usage']['prompt_tokens'] + response['usage']['completion_tokens'])
        
    except Exception as e:
        print("Error handled with retry:", e)

asyncio.run(non_logprobs_example())
```

## Scripts

- `scripts/104_search_all_openrouter_logprobs.ipynb`: Search and list OpenRouter models with logprobs (unchanged as is).
- `scripts/03_openrouter_topk.ipynb`: Example of logprobs for top-k choices.
- `scripts/03_judge_openrouter_CoT.py`: Example of structured judging with Chain of Thought.

## Notes

- All examples are async; use `asyncio.run()` to execute.
- For sync compatibility, wrap in executor if needed, but async is recommended.
- Provider whitelists/blocklists help select logprobs-capable endpoints.
- See source files for detailed implementation (e.g., `retry.py` for error handling).
