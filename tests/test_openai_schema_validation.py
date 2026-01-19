"""Validate that generated schemas meet OpenAI's requirements."""
import json
import sys
from pathlib import Path

from chat_factory import extract_function_schema
# pyright: reportMissingImports=false
from to_do import ToDo

# Add examples directory to path for to_do import
examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


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


def test_openai_schema_validation():
    """Test that generated schemas meet OpenAI requirements."""
    print("=" * 80)
    print("OpenAI Schema Validation Test")
    print("=" * 80)

    todo = ToDo()
    schema = extract_function_schema(todo.create_todos)

    print("\nGenerated schema for create_todos:")
    print(json.dumps(schema, indent=2))

    print("\n" + "=" * 80)
    print("Validating OpenAI Requirements...")
    print("=" * 80)

    # Validate the array parameter
    validate_openai_array_schema(schema, "descriptions")
    print("\n✅ Schema validation PASSED!")
    print("✅ Array parameter includes required 'items' property")
    print("✅ OpenAI API will accept this schema")

    # Show what was fixed
    descriptions_schema = schema["parameters"]["properties"]["descriptions"]
    print(f"\n✅ Type: {descriptions_schema['type']}")
    print(f"✅ Items type: {descriptions_schema['items']['type']}")
    print(f"✅ Description: {descriptions_schema['description'][:50]}...")

    print("\n" + "=" * 80)
    print("SUCCESS: Schema is OpenAI-compatible!")
    print("=" * 80)
