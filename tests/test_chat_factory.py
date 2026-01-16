#!/usr/bin/env python3
"""Test script for ChatFactory with MCP integration."""
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from chat_factory import ChatFactory, ChatModel
# pyright: reportMissingImports=false
from utils.to_do import ToDo

load_dotenv(find_dotenv(), override=True)

examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))


# Create test tools
to_do = ToDo()
tools = [to_do.get_todo_report, to_do.create_todos, to_do.mark_complete, to_do.clear_todos]

system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Markdown markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""


def test_chat_factory_creation():
    """Test ChatFactory creation with tools."""
    print("Creating ChatModel...")
    openai_model = ChatModel(model_name="gpt-4o-mini", provider="openai")

    print("Creating ChatFactory...")
    factory = ChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
        tools=tools,
        generator_kwargs={},
    )
    print("✅ ChatFactory created successfully!")
    print(f"   - MCP Client initialized: {factory.mcp_client is not None}")
    print(f"   - Total tools registered: {len(factory.openai_tools)}")

    # Verify basic properties
    assert factory.openai_tools is not None
    assert len(factory.openai_tools) > 0
    assert len(factory.openai_tools) == len(tools)

    print("\n✅ ChatFactory is ready to use!")
    print(f"   - Tools registered: {len(factory.openai_tools)}")
