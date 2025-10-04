"""
When using thinking models, we often can't use structured outputs, so we use soft structurings, asking for a simple outputs structure and trying json and/or regexp
"""
from loguru import logger
import json 
import re

def json_decode(s):
    if '{' not in s or '}' not in s:
        raise ValueError(f"Invalid JSON-like string: {s[-100:]}")
    s = '{'+s.split('{', 1)[1]
    s = s[::-1].split('}', 1)[1][::-1]+'}'
    return json.loads(s)



def try_to_regex_decode(s, allowed_keys=[]):
    ratings = {}
    matches = re.findall(r'["\']?(\w+)["\']?\s*:\s*["\']?(\d+)["\']?', s)
    for key, val in matches:
        key = key.lower()
        if len(allowed_keys):
            if key in allowed_keys:
                ratings[key] = int(val)
        else:
            ratings[key] = int(val)

    if len(allowed_keys):
        assert len(ratings) == len(allowed_keys), f"Expected {len(allowed_keys)} ratings, got {len(ratings)}"
    else:
        assert len(ratings) > 0, f"No ratings found in {s[-100:]}"
    return ratings


class DecodeLLMError(Exception):
    pass

def extract_answer_from_json(s, allowed_keys):

    exceptions = []
    try:
        return json_decode(s)
    except (json.JSONDecodeError, ValueError) as e1:
        logger.debug(f"Failed to decode JSON, trying regex {e1}")
        exceptions.append(e1)
        try:
            return try_to_regex_decode(s, allowed_keys)
        except AssertionError as e2:
            logger.exception(f"Failed to decode with regex {e2}")
            exceptions.append(e2)
    raise DecodeLLMError(f"Unable to decode answer: \nexceptions={exceptions}\n{s}")


# import random
# from .retry import openrouter_request

# def structured_openrouter_req(model_id, messages, provider_whitelist=None, **kwargs):
#     """
#     Here the messages should ask for a simple json output format
#     """
#     for i in range(2):
#         seed = random.randint(0, 2**30) + i
#         try:
#             payload={
#                 "model": model_id,
#                 "messages": messages,
#                 "provider": {"require_parameters": True, },
#                 "usage": {"include": True},
#                 "provider_whitelist": provider_whitelist,
#                 "reasoning": {"include": True},
#                 # "logprobs": True,
#             # "top_logprobs": 5,
#                 "max_completion_tokens": 4_000,
#                 "temperature": 0.5, # same as judgebench
#                 "top_k": 3, # same as judgebench
#                 "seed": seed,
#                 **kwargs
#             }

#             r = openrouter_request(payload)
#             answer = '<think>' + (r['choices'][0]['message']['reasoning'] or "") + '</think>\n' + (r['choices'][0]['message']['content'] or "")
#             rating_dict = extract_answer_from_json(answer, task_names)
#         except DecodeLLMError as e:
#             logger.warning(f"DecodeLLMError, retrying once: {e}")
#         else:
#             break
#     return r, rating_dict



# def swap_in_logprobs(r):
#     """Could also ask for logprobs, but it would restrict providers
#     Then we can swap in the whole logprob dist as a json, in place of single int rating
#     """
#     tokens = []
#     for row in r['choices'][0]['logprobs']['content']:
#         # tokens.append(row['token'])
#         try:
#             i = int(row['token'])
#             lp = row['top_logprobs']
#             lp = {x['token']: x['logprob'] for x in lp}
#             print(i, lp)
#             tokens.append(json.dumps(lp))
#         except ValueError:
#             tokens.append(row['token'])
#             pass
#     return tokens


def flatten_paydantic_schema(schema):
    """Remove $defs and inline all references"""
    if '$defs' not in schema:
        return schema
    
    defs = schema.pop('$defs')
    
    def replace_refs(obj):
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref_path = obj['$ref'].split('/')[-1]  # Get last part after '#/$defs/'
                return defs[ref_path]
            else:
                return {k: replace_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_refs(item) for item in obj]
        else:
            return obj
    
    return replace_refs(schema)
