import re
from . import OPENROUTER_API_KEY
from typing import Dict, List, Optional
from loguru import logger
from collections import OrderedDict
import requests
import numpy as np
import polars as pl
import stamina
import httpx
from anycache import anycache

from .retry import RetryException, retry_only_on_real_errors, ProviderError, UpstreamError, LogprobsNotSupportedError, MalformedResponseError


def get_top_logprobs_param(model_id, provider):
    """
    Different providers have different limits on the number of top logprobs they return.


        {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"error":{"object":"error","type":"invalid_request_error","message":"logprobs must be between 0 and 5: 20"}}', 'provider_name': 'Fireworks'}}}

        {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"code":"Client specified an invalid argument","error":"top_logprobs must be less equal than 8 but top_logprobs = 20"}', 'provider_name': 'xAI'}},}

    """
    # this should be via provider actually?
    if model_id.startswith('x-ai'):
        return 8
    if provider == 'Fireworks':
        return 5
    return 20

@anycache(cachedir="../.anycache3")
@stamina.retry(on=retry_only_on_real_errors, attempts=5, wait_max=20)
def openrouter_completion_wlogprobs(messages, model_id, max_completion_tokens=2, provider_whitelist=None, provider_blocklist=None, **kwargs):
    """"Run a completion with logprobs on OpenRouter."""
    response = httpx.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,

            # openai wants an int here
            "logprobs": True,

            # grok wants 8
            "top_logprobs": get_top_logprobs_param(model_id, provider_whitelist[0] if provider_whitelist else None),
            "max_completion_tokens": max_completion_tokens,
            # "temperature": 0.0,
            "stop": '</ans>',
            "provider": {"require_parameters": True, 
                         "only": provider_whitelist,
                         'ignore': provider_blocklist
                         },
            "usage": {"include": True},
            "reason": {"include": True, "effort": "low", 
                       "max_tokens": 0
                       },

            # "logit_bias": {
            #     " ": -100,
            #     "\n": -100,
            #     "": -100,
            # },
            **kwargs,

        },
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


def get_logprobs(response, regex: Optional[str] = '\d'):
    """Get the logprobs from the response, get it from the first token that satifies the regex."""
    try:
        content = response['choices'][0]['logprobs']['content']
    except KeyError:
        content = response['choices'][0]['logprobs']['top_logprobs']
        logger.warning('non openai logprobs fmt')

    assert len(content) > 0, f"no content in {response}"
    for i in range(len(content)):
        s = "".join([t['token'] for t in content[:i]])
        if re.search(regex, s):
            # logger.warning(f"Found matching token sequence: {s}")
            break

    lps = content[i-1]['top_logprobs']
    

    # lps = content[0]["top_logprobs"]
    # other times
    logp_dict = {}
    for lp in lps:
        t = lp["token"].lower()
        if t not in logp_dict:
            logp_dict[t] = lp["logprob"]
        else:
            logp_dict[t] += lp["logprob"]
    return logp_dict

def get_logprobs_choices(response, completion_tokens: list):
    """Get the logprobs from the response and permute them to match the completion tokens."""
    logp_dict = get_logprobs(response)
    choice_logp_dict = OrderedDict({t: logp_dict.get(t, -1000.) for t in completion_tokens})
    return choice_logp_dict, logp_dict

def get_logprobs_numchoices(response, num_choices: int):
    """Get the logprobs from the response and permute them to match the number of choices."""
    logp_dict, logp_dict_all = get_logprobs_choices(response, completion_tokens=[str(i) for i in range(1, num_choices + 1)])
    completion_tokens = list(logp_dict.keys())[:num_choices]
    choice_logp_arr = np.array([[int(t), logp_dict.get(t, -1000.)] for t in completion_tokens])

    # sort first axis
    choice_logp_arr = choice_logp_arr[np.argsort(choice_logp_arr[:, 0])]

    # take just the ordered choices
    choice_logp_arr = choice_logp_arr[:, 1]
    return choice_logp_arr, logp_dict_all


def eval_message_choice_logp(messages: List[Dict[str, str]], model_id: str, provider_whitelist: Optional[List[str]] = None, **kwargs) -> tuple:
    response_data = openrouter_completion_wlogprobs(messages, model_id, provider_whitelist=provider_whitelist, **kwargs)
    num_actions = 5
    choice_logp_arr, logp_dict = get_logprobs_numchoices(response_data, num_actions)
    return choice_logp_arr, response_data, logp_dict
