"""Test that ToDo class methods can be registered as OpenAI tools."""

import json
import sys
from pathlib import Path

from chat_factory.utils.factory_utils import convert_tools_to_openai_format
# pyright: reportMissingImports=false
from utils.to_do import ToDo

# Add examples directory to path for to_do import
examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


def test_todo_tool_registration():
    """Test complete tool registration flow for ToDo methods."""
    # Create ToDo instance
    todo = ToDo()

    # Register all public methods as tools
    tools = [
        todo.get_todo_report,
        todo.create_todos,
        todo.mark_complete,
        todo.clear_todos,
    ]

    print("=" * 80)
    print("Testing ToDo Tool Registration with convert_tools_to_openai_format")
    print("=" * 80)

    # Convert to OpenAI format
    openai_tools, tool_map = convert_tools_to_openai_format(tools)

    print(f"\n✓ Successfully converted {len(openai_tools)} tools")
    print(f"✓ Tool map contains {len(tool_map)} function references\n")

    # Verify tool map
    expected_names = ["get_todo_report", "create_todos", "mark_complete", "clear_todos"]
    for name in expected_names:
        assert name in tool_map, f"Missing {name} in tool_map"
        assert callable(tool_map[name]), f"{name} is not callable"
        print(f"✓ Tool map contains callable: {name}")

    # Display all generated tools
    print("\n" + "=" * 80)
    print("Generated OpenAI Tools:")
    print("=" * 80)

    for i, tool in enumerate(openai_tools, 1):
        print(f"\n--- Tool {i}: {tool['function']['name']} ---")
        print(json.dumps(tool, indent=2))

    # Test that tools work when called via tool_map
    print("\n" + "=" * 80)
    print("Testing Tool Execution:")
    print("=" * 80)

    # Test create_todos
    print("\n1. Testing create_todos...")
    result = tool_map["create_todos"](descriptions=["Write docs", "Write tests"])
    print(f"   Result: {result.strip()}")

    # Test get_todo_report
    print("\n2. Testing get_todo_report...")
    result = tool_map["get_todo_report"]()
    print(f"   Result: {result.strip()}")

    # Test mark_complete
    print("\n3. Testing mark_complete...")
    result = tool_map["mark_complete"](index=1, completion_notes="Documentation complete!")
    print(f"   Result: {result.strip()}")

    # Test clear_todos
    print("\n4. Testing clear_todos...")
    result = tool_map["clear_todos"]()
    print(f"   Result: {result.strip()}")

    print("\n" + "=" * 80)
    print("✓ All tools registered and executed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_todo_tool_registration()
