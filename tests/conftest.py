"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest


THIS_DIR = Path(__file__).parent
TESTS_DIR_PARENT = (THIS_DIR / "..").resolve()

# Ensure that `from tests ...` import statements work within the tests/ dir
sys.path.insert(0, str(TESTS_DIR_PARENT))

# Add src directory to path to ensure package can be imported
src_dir = TESTS_DIR_PARENT / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Add test directory to path
tests_dir = Path(__file__).parent.parent / "tests"
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


# ============================================================================
# Mock ChatModel Fixtures
# ============================================================================


@pytest.fixture
def mock_chat_model():
    """Mock ChatModel that returns controlled responses."""
    model = MagicMock()
    model.generate_response.return_value = "Test response"
    model.format_tool_result.return_value = {"role": "tool", "content": "result", "tool_call_id": "call_1"}
    model.stream_response.return_value = iter(["chunk1", "chunk2", "chunk3"])
    return model


@pytest.fixture
def mock_chat_model_with_tool_calls():
    """Mock ChatModel that returns tool calls then final response."""
    model = MagicMock()

    # Create mock tool call
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = '{"city": "London"}'

    # First call returns tool calls (list), second returns final response
    model.generate_response.side_effect = [[tool_call], "Final response after tool"]
    model.format_tool_result.return_value = {"role": "tool", "content": "sunny", "tool_call_id": "call_123"}
    return model


@pytest.fixture
def mock_evaluator_model():
    """Mock evaluator model that returns Evaluation responses."""
    from chat_factory.utils.factory import Evaluation

    model = MagicMock()
    model.generate_response.return_value = Evaluation(is_acceptable=True, feedback="")
    return model


@pytest.fixture
def mock_evaluator_model_rejects_then_accepts():
    """Mock evaluator that rejects first, then accepts."""
    from chat_factory.utils.factory import Evaluation

    model = MagicMock()
    model.generate_response.side_effect = [
        Evaluation(is_acceptable=False, feedback="Response too short"),
        Evaluation(is_acceptable=True, feedback="Good response"),
    ]
    return model


# ============================================================================
# Mock Async ChatModel Fixtures
# ============================================================================


@pytest.fixture
def mock_async_chat_model():
    """Mock AsyncChatModel with async methods."""
    model = MagicMock()
    model.agenerate_response = AsyncMock(return_value="Async test response")
    model.format_tool_result.return_value = {"role": "tool", "content": "result", "tool_call_id": "call_1"}

    async def mock_stream():
        for chunk in ["async", " ", "stream"]:
            yield chunk

    model.astream_response = MagicMock(return_value=mock_stream())
    return model


@pytest.fixture
def mock_async_chat_model_with_tool_calls():
    """Mock AsyncChatModel that returns tool calls then final response."""
    model = MagicMock()

    # Create mock tool call
    tool_call = MagicMock()
    tool_call.id = "call_456"
    tool_call.function.name = "calculate"
    tool_call.function.arguments = '{"a": 5, "b": 3}'

    # First call returns tool calls (list), second returns final response
    model.agenerate_response = AsyncMock(side_effect=[[tool_call], "Async final response after tool"])
    model.format_tool_result.return_value = {"role": "tool", "content": "8", "tool_call_id": "call_456"}
    return model


@pytest.fixture
def mock_async_evaluator_model():
    """Mock async evaluator model that returns Evaluation responses."""
    from chat_factory.utils.factory import Evaluation

    model = MagicMock()
    model.agenerate_response = AsyncMock(return_value=Evaluation(is_acceptable=True, feedback=""))
    return model


# ============================================================================
# Mock MCP Client Fixtures
# ============================================================================


@pytest.fixture
def mock_mcp_client():
    """Mock SyncMultiServerClient for testing MCP integration."""
    client = MagicMock()

    # Mock list_tools
    tools_result = MagicMock()
    tools_result.tools = []
    client.list_tools.return_value = tools_result

    # Mock list_prompts
    prompts_result = MagicMock()
    prompts_result.prompts = []
    client.list_prompts.return_value = prompts_result

    # Mock list_resources
    resources_result = MagicMock()
    resources_result.resources = []
    client.list_resources.return_value = resources_result

    # Mock list_resource_templates
    templates_result = MagicMock()
    templates_result.resourceTemplates = []
    client.list_resource_templates.return_value = templates_result

    client.shutdown = MagicMock()
    client.call_tool = MagicMock()
    client.get_prompt = MagicMock()
    client.read_resource = MagicMock()
    client.set_logging_level = MagicMock()

    return client


@pytest.fixture
def mock_async_mcp_client():
    """Mock MultiServerClient for async testing."""
    client = MagicMock()

    # Mock list_tools
    tools_result = MagicMock()
    tools_result.tools = []
    client.list_tools.return_value = tools_result

    # Mock list_prompts
    prompts_result = MagicMock()
    prompts_result.prompts = []
    client.list_prompts.return_value = prompts_result

    # Mock list_resources
    resources_result = MagicMock()
    resources_result.resources = []
    client.list_resources.return_value = resources_result

    # Mock list_resource_templates
    templates_result = MagicMock()
    templates_result.resourceTemplates = []
    client.list_resource_templates.return_value = templates_result

    # Async methods
    client.connect_all = AsyncMock()
    client.call_tool = AsyncMock()
    client.get_prompt = AsyncMock()
    client.read_resource = AsyncMock()
    client.set_logging_level = AsyncMock()

    return client


# ============================================================================
# Sample Tools Fixtures
# ============================================================================


@pytest.fixture
def sample_tools():
    """Sample tool functions for testing."""

    def get_weather(city: str) -> dict:
        """Get weather for a city.

        Args:
            city: The city name
        """
        return {"temp": 72, "condition": "sunny"}

    def calculate(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a + b

    return [get_weather, calculate]


@pytest.fixture
def sample_tool_dict():
    """Sample tool with dict format (custom description)."""

    def my_tool(x: str) -> str:
        """Process a string input.

        Args:
            x: The input string to process
        """
        return f"processed: {x}"

    return {"function": my_tool, "description": "Custom description for my_tool"}


@pytest.fixture
def sample_history() -> List[Dict[str, Any]]:
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "user", "content": "What's the weather?"},
    ]


@pytest.fixture
def empty_history() -> List[Dict[str, Any]]:
    """Empty conversation history."""
    return []
