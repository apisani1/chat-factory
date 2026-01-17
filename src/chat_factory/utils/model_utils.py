"""Shared utility functions for ChatModel and AsyncChatModel."""

import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


OPENAI_CLIENT_MAP = {
    "openai": {"base_url": "https://api.openai.com/v1", "env_var": "OPENAI_API_KEY"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env_var": "GOOGLE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env_var": "DEEPSEEK_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_var": "GROQ_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "env_var": "OLLAMA_API_KEY"},
}


def prepare_messages_for_anthropic(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Prepare messages for Anthropic API by:
    1. Extracting system messages to separate parameter
    2. Converting OpenAI tool_calls format to Anthropic content blocks

    Anthropic's API differences from OpenAI:
    - System messages must be passed via separate 'system' parameter
    - Assistant tool calls use content blocks, not tool_calls field
    - Tool calls are dicts: {"type": "tool_use", "id", "name", "input"}

    Args:
        messages: List of message dictionaries (OpenAI format)

    Returns:
        tuple: (system_content, anthropic_messages)
            - system_content: Combined system message string or None
            - anthropic_messages: Messages list with Anthropic-specific conversions
    """
    # Step 1: Extract system messages (existing behavior)
    system_messages = []
    remaining_messages = messages

    while remaining_messages and remaining_messages[0].get("role") == "system":
        system_messages.append(remaining_messages[0]["content"])
        remaining_messages = remaining_messages[1:]

    system_content = "\n\n".join(system_messages) if system_messages else ""

    # Step 2: Convert tool_calls to Anthropic format
    anthropic_messages = []
    for msg in remaining_messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            # Convert OpenAI tool_calls to Anthropic content blocks
            tool_use_blocks = []
            for tool_call in msg["tool_calls"]:
                tool_use_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments),
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": tool_use_blocks})
        else:
            # Keep other messages as-is
            anthropic_messages.append(msg)

    return system_content, anthropic_messages


def prepare_tool_params(response_format: Optional[type[BaseModel]]) -> Dict[str, Any]:
    """
    Prepare tool parameters for Anthropic's structured response via tool use.

    Args:
        response_format: Pydantic model defining the expected response structure

    Returns:
        dict: Tool parameters for the API call, or empty dict if not using structured response
    """
    if response_format is None:
        return {}

    return {
        "tools": [
            {
                "name": "structured_response",
                "description": "Return a structured response",
                "input_schema": response_format.model_json_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": "structured_response"},
    }


def convert_tools_to_anthropic(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Convert OpenAI tool format to Anthropic format.

    OpenAI format: {"type": "function", "function": {name, description, parameters}}
    Anthropic format: {name, description, input_schema}

    Args:
        tools: List of tool definitions in OpenAI format

    Returns:
        List of tool definitions in Anthropic format
    """
    if not tools:
        return {}

    anthropic_tools = []
    for tool in tools:
        # Handle both full format and just the function definition
        func = tool.get("function", tool)
        anthropic_tools.append(
            {
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"],
            }
        )
    return {"tools": anthropic_tools}


def convert_tool_calls_to_openai(tool_use_blocks: List[Any]) -> List[ChatCompletionMessageToolCall]:
    """
    Convert Anthropic tool_use blocks to OpenAI tool_calls format.

    Anthropic format: ToolUseBlock(id, type="tool_use", name, input)
    OpenAI format: ChatCompletionMessageToolCall objects

    Args:
        tool_use_blocks: List of Anthropic ToolUseBlock objects

    Returns:
        List[ChatCompletionMessageToolCall]: Tool calls in OpenAI format
    """
    openai_tool_calls = []
    for tool_use in tool_use_blocks:
        function = Function(name=tool_use.name, arguments=json.dumps(tool_use.input))
        tool_call = ChatCompletionMessageToolCall(id=tool_use.id, type="function", function=function)
        openai_tool_calls.append(tool_call)
    return openai_tool_calls


def format_tool_result(provider: str, tool_call_id: str, result: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Format tool execution result for adding back to conversation.

    Handles provider-specific formatting differences between OpenAI and Anthropic.

    Args:
        provider: The provider name ("openai", "anthropic", etc.)
        tool_call_id: ID of the tool call from the response
        result: Result from tool execution (dict or str)

    Returns:
        dict: Formatted message to append to conversation history

    Examples:
        >>> result_msg = format_tool_result(
        ...     "openai",
        ...     tool_call_id="call_123",
        ...     result={"temp": 72, "condition": "sunny"}
        ... )
        >>> messages.append(result_msg)
    """
    content = json.dumps(result) if isinstance(result, dict) else str(result)

    if provider in OPENAI_CLIENT_MAP:
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }

    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    raise ValueError(f"Unsupported provider: {provider}")
