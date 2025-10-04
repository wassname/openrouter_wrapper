import re
from . import OPENROUTER_API_KEY
from typing import Dict, List, Optional, Any
from loguru import logger
from collections import OrderedDict
import numpy as np
import httpx

from .retry import openrouter_request, LogprobsNotSupportedError


def get_top_logprobs_param(model_id: str, provider: Optional[str]) -> int:
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


async def openrouter_completion_wlogprobs(
    messages: List[Dict[str, Any]],
    model_id: str,
    max_completion_tokens: int = 2,
    provider_whitelist: Optional[List[str]] = None,
    provider_blocklist: Optional[List[str]] = None,
    stop: List[str] = [],
    **kwargs
) -> Dict[str, Any]:
    provider = provider_whitelist[0] if provider_whitelist else None
    json_payload = {
        "model": model_id,
        "messages": messages,
        # grok wants 8
        "logprobs": True,
        "top_logprobs": get_top_logprobs_param(model_id, provider),
        "max_completion_tokens": max_completion_tokens,
        # "temperature": 0.0,
        "stop": stop,
        "provider": {
            "require_parameters": True, 
            "only": provider_whitelist,
            'ignore': provider_blocklist
        },
        "usage": {"include": True},
        "reason": {"include": True, "effort": "low", 
                    "max_tokens": 60
                    },

        # "logit_bias": {
        #     " ": -100,
        #     "\n": -100,
        #     "": -100,
        # },
        **kwargs,
    }
    data = await openrouter_request(json_payload)

    if not data['choices'][0].get('logprobs'):
        raise LogprobsNotSupportedError(f"{model_id} has no logprobs capability", data=data)
    return data


def get_logprobs(
    response: Dict[str, Any], 
    regex: Optional[str] = '\d'
) -> Dict[str, float]:
    """Get the logprobs from the response, get it from the first token that satisfies the regex."""
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


def get_logprobs_choices(
    response: Dict[str, Any], 
    completion_tokens: List[str], 
    regex: str = '\d'
) -> tuple[OrderedDict[str, float], Dict[str, float]]:
    """Get the logprobs from the response and permute them to match the completion tokens."""
    logp_dict = get_logprobs(response, regex=regex)
    choice_logp_dict = OrderedDict({t: logp_dict.get(t, -1000.) for t in completion_tokens})
    return choice_logp_dict, logp_dict


def get_logprobs_numchoices(
    response: Dict[str, Any],
    num_choices: int
) -> tuple[np.ndarray, Dict[str, float]]:
    """Get the logprobs from the response and permute them to match the number of choices."""
    logp_dict, logp_dict_all = get_logprobs_choices(response, completion_tokens=[str(i) for i in range(1, num_choices + 1)])
    completion_tokens = list(logp_dict.keys())[:num_choices]
    choice_logp_arr = np.array([[int(t), logp_dict.get(t, -1000.)] for t in completion_tokens])

    # sort first axis
    choice_logp_arr = choice_logp_arr[np.argsort(choice_logp_arr[:, 0])]

    # take just the ordered choices
    choice_logp_arr = choice_logp_arr[:, 1]
    return choice_logp_arr, logp_dict_all


async def eval_message_choice_logp(
    messages: List[Dict[str, Any]],
    model_id: str,
    provider_whitelist: Optional[List[str]] = None,
    **kwargs
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
    response_data = await openrouter_completion_wlogprobs(messages, model_id, provider_whitelist=provider_whitelist, **kwargs)
    num_actions = 5
    choice_logp_arr, logp_dict = get_logprobs_numchoices(response_data, num_actions)
    return choice_logp_arr, response_data, logp_dict
