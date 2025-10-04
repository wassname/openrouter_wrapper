import os
from typing import Optional, Any
import polars as pl
import httpx


async def get_openrouter_models() -> pl.DataFrame:
    """List all models available on OpenRouter asynchronously."""
    url = "https://openrouter.ai/api/v1/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    df_models = pl.DataFrame.from_dict(data['data'])
    return df_models


async def get_logp_endpoints(model_id: str) -> tuple[pl.DataFrame, dict]:
    """Get endpoints with logprobs support for a model asynchronously."""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()['data']
    df_end = pl.from_records(data['endpoints'])

    df_end = df_end.with_columns(
        pl.col('supported_parameters').map_elements(lambda x: 'top_logprobs' in x, return_dtype=pl.Boolean).alias('top_logprobs'),
        pl.col('supported_parameters').map_elements(lambda x: 'logprobs' in x, return_dtype=pl.Boolean).alias('logprobs'),
        pl.col('pricing').map_elements(lambda x: x['prompt'], return_dtype=str).alias('price_prompt'),
    )
    df_end = df_end.with_columns(
        price_prompt=pl.col('price_prompt').cast(pl.Float64),
        uptime_last_30m=pl.col('uptime_last_30m').cast(pl.Float64)
    )

    df_end = df_end.filter(pl.col('top_logprobs'))
    df_end = df_end.filter(pl.col('logprobs'))
    df_end = df_end.sort('price_prompt', descending=True)
    return df_end, data
