import os
from typing import Optional
import requests
import polars as pl


def get_openrouter_models():
    """list all models available on OpenRouter"""
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    df_models = pl.DataFrame.from_dict(data['data'])
    return df_models

def get_logp_endpoints(model_id):
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()['data']
    df_end = pl.from_records(data['endpoints'])

    df_end = df_end.with_columns(
        pl.col('supported_parameters').map_elements(lambda x: 'top_logprobs' in x, return_dtype=pl.Boolean).alias('top_logprobs'),
        # 'top_logprobs'.is_in(pl.col("supported_parameters")).alias('top_logprobs'),
        # pl.col('supported_parameters').map_elements(lambda x: 'top_logprobs' in x, return_dtype=pl.Boolean).alias('top_logprobs'),
        pl.col('supported_parameters').map_elements(lambda x: 'logprobs' in x, return_dtype=pl.Boolean).alias('logprobs'),
        pl.col('pricing').map_elements(lambda x: x['prompt'],
                                    return_dtype=str).alias('price_prompt'),
    )
    df_end = df_end.with_columns(
        price_prompt=pl.col('price_prompt').cast(pl.Float64),
        uptime_last_30m=pl.col('uptime_last_30m').cast(pl.Float64)
    )

    # df_end['top_logprobs'] = df_end['supported_parameters'].apply(lambda x: 'top_logprobs' in x)
    # df_end['logprobs'] = df_end['supported_parameters'].apply(lambda x: 'logprobs' in x)
    # df_end['price_prompt'] = pl.col('pricing').apply(lambda x: x['prompt'])
    # for c in ['pricing', 'uptime_last_30m']:
    #     if c in df_end.columns:
    #         df_end[c] = pl.col(c).cast(pl.Float64)

    df_end = df_end.filter(pl.col('top_logprobs'))
    df_end = df_end.filter(pl.col('logprobs'))
    df_end = df_end.sort('price_prompt', descending=True)
    return df_end, data
