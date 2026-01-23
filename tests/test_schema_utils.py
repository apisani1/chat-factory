"""Comprehensive tests for schema utilities: generation, validation, and error handling."""

import warnings
from typing import List

import pytest

from chat_factory import (
    _map_python_type_to_json_schema,
    _parse_google_docstring,
    extract_function_schema,
)
from chat_factory.utils.factory import convert_tools_to_openai_format

# pyright: reportMissingImports=false
from to_do import ToDo


# ============================================================================
# Sample Functions for Testing
# ============================================================================


def sample_tool_func(email: str, name: str = "default"):
    """Test tool function.

    Args:
        email: User email address
        name: User name

    Returns:
        dict: Test result
    """
    return {"email": email, "name": name}


def valid_func(x: int):
    """Valid function.

    Args:
        x: An integer
    """
    return x


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


# ============================================================================
# Type Mapping Tests
# ============================================================================


class TestTypeMapping:
    """Tests for Python type to JSON Schema type mapping."""

    def test_string_type(self):
        """Test string type mapping."""
        assert _map_python_type_to_json_schema(str) == "string"

    def test_integer_type(self):
        """Test integer type mapping."""
        assert _map_python_type_to_json_schema(int) == "integer"

    def test_float_type(self):
        """Test float type mapping."""
        assert _map_python_type_to_json_schema(float) == "number"

    def test_boolean_type(self):
        """Test boolean type mapping."""
        assert _map_python_type_to_json_schema(bool) == "boolean"

    def test_list_type(self):
        """Test list type mapping."""
        assert _map_python_type_to_json_schema(list) == "array"

    def test_dict_type(self):
        """Test dict type mapping."""
        assert _map_python_type_to_json_schema(dict) == "object"


# ============================================================================
# Docstring Parsing Tests
# ============================================================================


class TestDocstringParsing:
    """Tests for Google-style docstring parsing."""

    def test_basic_docstring(self):
        """Test parsing a basic Google-style docstring."""
        docstring = """Test function.

        Args:
            param1: Description 1
            param2: Description 2
        """
        desc, params = _parse_google_docstring(docstring)
        assert desc == "Test function."
        assert params["param1"] == "Description 1"
        assert params["param2"] == "Description 2"

    def test_non_string_docstring_integer(self):
        """Test that integer docstrings are handled gracefully."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            desc, params = _parse_google_docstring(123)  # type: ignore
            assert desc == ""
            assert params == {}
            assert len(w) == 1
            assert "must be a string" in str(w[0].message)

    def test_non_string_docstring_list(self):
        """Test that list docstrings are handled gracefully."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            desc, params = _parse_google_docstring([1, 2, 3])  # type: ignore
            assert desc == ""
            assert params == {}
            assert len(w) == 1
            assert "must be a string" in str(w[0].message)

    def test_non_string_docstring_dict(self):
        """Test that dict docstrings are handled gracefully."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            desc, params = _parse_google_docstring({"key": "value"})  # type: ignore
            assert desc == ""
            assert params == {}
            assert len(w) == 1
            assert "must be a string" in str(w[0].message)

    def test_function_with_non_string_doc(self):
        """Test function with non-string __doc__ attribute."""

        def weird_func(x: int):
            pass

        # Manually set __doc__ to non-string (unusual but possible)
        weird_func.__doc__ = 12345  # type: ignore

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = extract_function_schema(weird_func)

            # Should still generate schema, just without description
            assert schema["name"] == "weird_func"
            assert "x" in schema["parameters"]["properties"]

            # Should have warned about non-string docstring
            warning_messages = [str(warning.message) for warning in w]
            assert any("must be a string" in msg for msg in warning_messages)


# ============================================================================
# Schema Extraction Tests
# ============================================================================


class TestSchemaExtraction:
    """Tests for schema extraction from functions."""

    def test_basic_schema_extraction(self):
        """Test schema extraction from function."""
        schema = extract_function_schema(sample_tool_func)

        assert schema["name"] == "sample_tool_func"
        assert schema["description"] == "Test tool function."
        assert schema["parameters"]["required"] == ["email"]
        assert schema["parameters"]["properties"]["email"]["type"] == "string"
        assert schema["parameters"]["properties"]["name"]["type"] == "string"

    def test_format_1_just_function(self):
        """Test Format 1: Just passing functions."""
        tools = [sample_tool_func]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "sample_tool_func"
        assert "email" in openai_tools[0]["function"]["parameters"]["properties"]
        assert "sample_tool_func" in tool_map
        assert callable(tool_map["sample_tool_func"])

    def test_format_2_dict_with_autogen(self):
        """Test Format 2: Dict with auto-generation."""
        tools = [{"function": sample_tool_func, "description": "Custom description override"}]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore

        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["description"] == "Custom description override"
        assert openai_tools[0]["function"]["name"] == "sample_tool_func"
        assert "email" in openai_tools[0]["function"]["parameters"]["properties"]

    def test_format_3_manual_schema(self):
        """Test Format 3: Manual schema (backward compatible)."""
        manual_schema = {
            "function": sample_tool_func,
            "description": "Manual description",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Manual email desc"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        }
        tools = [manual_schema]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore

        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["description"] == "Manual description"
        assert openai_tools[0]["function"]["parameters"] == manual_schema["parameters"]

    def test_mixed_formats(self):
        """Test mixing all three formats."""

        def another_func(x: int):
            """Another test function.

            Args:
                x: An integer

            Returns:
                int: Result
            """
            return x * 2

        manual_tool = {
            "function": sample_tool_func,
            "description": "Manual",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        tools = [
            another_func,  # Format 1
            {"function": sample_tool_func},  # Format 2
            manual_tool,  # Format 3
        ]

        openai_tools, tool_map = convert_tools_to_openai_format(tools)

        assert len(openai_tools) == 3
        assert openai_tools[0]["function"]["name"] == "another_func"  # Format 1
        assert openai_tools[1]["function"]["name"] == "sample_tool_func"  # Format 2
        assert openai_tools[2]["function"]["description"] == "Manual"  # Format 3


# ============================================================================
# Array Schema Edge Cases
# ============================================================================


class TestArraySchemaEdgeCases:
    """Tests for array schema generation edge cases."""

    def test_simple_string_array(self):
        """Test simple List[str] generates correct schema."""
        schema = extract_function_schema(simple_array_func)
        param_schema = schema["parameters"]["properties"]["items"]

        assert param_schema["type"] == "array"
        assert "items" in param_schema

    def test_integer_array(self):
        """Test List[int] generates correct schema."""
        schema = extract_function_schema(integer_array_func)
        param_schema = schema["parameters"]["properties"]["numbers"]

        assert param_schema["type"] == "array"
        assert "items" in param_schema

    def test_nested_array(self):
        """Test List[List[int]] generates correct schema."""
        schema = extract_function_schema(nested_array_func)
        param_schema = schema["parameters"]["properties"]["matrix"]

        assert param_schema["type"] == "array"
        assert "items" in param_schema

    def test_triple_nested_array(self):
        """Test List[List[List[str]]] generates correct schema."""
        schema = extract_function_schema(triple_nested_func)
        param_schema = schema["parameters"]["properties"]["data"]

        assert param_schema["type"] == "array"
        assert "items" in param_schema


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in schema generation."""

    def test_error_missing_function_key(self):
        """Test error when dict has no 'function' key."""
        tools = [{"description": "oops, no function!"}]

        with pytest.raises(ValueError) as exc_info:
            convert_tools_to_openai_format(tools)  # type: ignore

        assert "must have 'function' key" in str(exc_info.value)

    def test_error_function_not_callable(self):
        """Test error when 'function' value is not callable."""
        tools = [{"function": "not_a_function"}]

        with pytest.raises(TypeError) as exc_info:
            convert_tools_to_openai_format(tools)  # type: ignore

        assert "must be callable" in str(exc_info.value)

    def test_error_tool_not_dict_or_callable(self):
        """Test error when tool is neither dict nor callable."""
        tools = [123]  # Integer instead of function/dict

        with pytest.raises(TypeError) as exc_info:
            convert_tools_to_openai_format(tools)  # type: ignore

        assert "must be a callable or dict" in str(exc_info.value)

    def test_error_function_none(self):
        """Test error when 'function' key exists but is None."""
        tools = [{"function": None}]

        with pytest.raises(ValueError) as exc_info:
            convert_tools_to_openai_format(tools)  # type: ignore

        assert "must have 'function' key" in str(exc_info.value)

    def test_valid_format_1_still_works(self):
        """Verify Format 1 (just function) still works."""
        tools = [valid_func]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore
        assert len(openai_tools) == 1

    def test_valid_format_2_still_works(self):
        """Verify Format 2 (dict with auto-gen) still works."""
        tools = [{"function": valid_func}]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore
        assert len(openai_tools) == 1

    def test_valid_format_3_still_works(self):
        """Verify Format 3 (dict with manual schema) still works."""
        tools = [
            {
                "function": valid_func,
                "description": "Manual",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        openai_tools, tool_map = convert_tools_to_openai_format(tools)  # type: ignore
        assert len(openai_tools) == 1


# ============================================================================
# OpenAI Schema Validation Tests
# ============================================================================


def validate_openai_array_schema(schema, param_name):
    """Validate that an array parameter meets OpenAI requirements.

    OpenAI requires array types to have an 'items' property.
    """
    param_schema = schema["parameters"]["properties"].get(param_name)

    if not param_schema:
        raise ValueError(f"Parameter '{param_name}' not found in schema")

    if param_schema.get("type") != "array":
        raise ValueError(f"Parameter '{param_name}' is not an array type")

    if "items" not in param_schema:
        raise ValueError(
            f"Parameter '{param_name}' is missing required 'items' property. "
            "OpenAI will reject this schema with error: "
            "'array schema missing items'"
        )

    items = param_schema["items"]
    if not isinstance(items, dict):
        raise ValueError(f"Parameter '{param_name}' items must be a dict, got {type(items)}")

    if "type" not in items:
        raise ValueError(f"Parameter '{param_name}' items missing 'type' property")

    return True


class TestOpenAISchemaValidation:
    """Tests that generated schemas meet OpenAI's requirements."""

    def test_todo_create_schema_is_valid(self):
        """Test that ToDo.create_todos generates OpenAI-compatible schema."""
        todo = ToDo()
        schema = extract_function_schema(todo.create_todos)

        # Validate the array parameter meets OpenAI requirements
        assert validate_openai_array_schema(schema, "descriptions") is True

        # Verify specific properties
        descriptions_schema = schema["parameters"]["properties"]["descriptions"]
        assert descriptions_schema["type"] == "array"
        assert "items" in descriptions_schema
        assert "type" in descriptions_schema["items"]
