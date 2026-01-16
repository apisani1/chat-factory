"""Test script to verify ToDo class methods generate correct OpenAI tool schemas."""

import json

from chat_factory import extract_function_schema
# pyright: reportMissingImports=false
from utils.to_do import ToDo


def test_todo_schemas():
    """Test that all ToDo methods generate proper schemas."""
    todo = ToDo()

    # Test all public methods
    methods = [
        ("get_todo_report", todo.get_todo_report),
        ("create_todos", todo.create_todos),
        ("mark_complete", todo.mark_complete),
        ("clear_todos", todo.clear_todos),
    ]

    print("=" * 80)
    print("Testing ToDo class method schema generation")
    print("=" * 80)

    for method_name, method in methods:
        print(f"\n\n{'='*80}")
        print(f"Schema for: {method_name}")
        print("=" * 80)

        schema = extract_function_schema(method)
        print(json.dumps(schema, indent=2))

        # Verify required fields are present
        assert "name" in schema, f"Missing 'name' in {method_name} schema"
        assert "description" in schema, f"Missing 'description' in {method_name} schema"
        assert "parameters" in schema, f"Missing 'parameters' in {method_name} schema"

        print(f"\n✓ Schema for {method_name} is valid!")

    print("\n" + "=" * 80)
    print("All schemas generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_todo_schemas()
