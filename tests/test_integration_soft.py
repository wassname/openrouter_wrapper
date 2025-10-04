import pytest
from openrouter_wrapper.soft_structure import extract_answer_from_json, flatten_paydantic_schema


def test_extract_answer_from_json():
    """Basic check: parse JSON and regex examples."""
    # JSON case
    content = '{"rating": 7}'
    result = extract_answer_from_json(content, ['rating'])
    assert 'rating' in result
    assert result['rating'] == 7

    # Regex fallback
    content = 'rating: 8'
    result = extract_answer_from_json(content, ['rating'])
    assert 'rating' in result
    assert result['rating'] == 8


def test_flatten_paydantic_schema():
    """Basic check: flatten schema with refs."""
    schema = {
        "$defs": {"RefType": {"type": "string"}},
        "properties": {"key": {"$ref": "#/$defs/RefType"}}
    }
    flattened = flatten_paydantic_schema(schema)
    assert '$defs' not in flattened
    assert flattened["properties"]["key"] == {"type": "string"}
