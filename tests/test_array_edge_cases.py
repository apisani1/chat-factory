"""Test edge cases for array schema generation."""

import json
from typing import List
from chat_factory import extract_function_schema


def simple_array_func(items: List[str]) -> str:
    """Test function with simple array parameter.

    Args:
        items: List of string items

    Returns:
        str: Result
    """
    return "ok"


def integer_array_func(numbers: List[int]) -> str:
    """Test function with integer array parameter.

    Args:
        numbers: List of integers

    Returns:
        str: Result
    """
    return "ok"


def nested_array_func(matrix: List[List[int]]) -> str:
    """Test function with nested array parameter.

    Args:
        matrix: 2D matrix of integers

    Returns:
        str: Result
    """
    return "ok"


def triple_nested_func(data: List[List[List[str]]]) -> str:
    """Test function with triple-nested array.

    Args:
        data: 3D array of strings

    Returns:
        str: Result
    """
    return "ok"


def test_array_edge_cases():
    """Test that array schema generation handles various edge cases."""
    print("=" * 80)
    print("Testing Array Schema Edge Cases")
    print("=" * 80)

    tests = [
        ("Simple Array (List[str])", simple_array_func, "items"),
        ("Integer Array (List[int])", integer_array_func, "numbers"),
        ("Nested Array (List[List[int]])", nested_array_func, "matrix"),
        ("Triple Nested (List[List[List[str]]])", triple_nested_func, "data"),
    ]

    for test_name, test_func, param_name in tests:
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print('='*80)

        schema = extract_function_schema(test_func)
        param_schema = schema["parameters"]["properties"][param_name]

        print(json.dumps(param_schema, indent=2))

        # Verify it has items property
        assert "items" in param_schema, f"Missing 'items' in {test_name}"
        assert param_schema["type"] == "array", f"Wrong type in {test_name}"

        print(f"\n✓ {test_name} generates valid schema with items property!")

    print("\n" + "=" * 80)
    print("All edge case tests passed!")
    print("=" * 80)
