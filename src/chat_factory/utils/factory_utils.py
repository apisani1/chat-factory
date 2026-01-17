"""Shared utilities for ChatFactory and AsyncChatFactory."""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

from .schema_utils import extract_function_schema


class Evaluation(BaseModel):
    """Result of evaluating an agent response."""

    is_acceptable: bool
    feedback: str


GENERATOR_PROMPT = """You are a helpful AI assistant.
Your responsibility is to provide accurate, professional, and engaging responses to user questions.
Be clear and concise in your answers.
If you don't know the answer to something, say so honestly rather than making up information."""

EVALUATOR_PROMPT = """You are an evaluator that decides whether a response to a question is acceptable quality.
You are provided with a conversation between a User and an Agent.
Your task is to decide whether the Agent's latest response is acceptable.
The Agent should be helpful, accurate, professional, and appropriate.
Consider whether the response:
- Accurately addresses the user's question
- Is professional and well-written
- Is complete and helpful
- Avoids making up information or being misleading
Please evaluate the latest response and provide feedback."""


def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove extra fields from messages that may be added by Gradio or other UIs.

    Args:
        messages: List of message dictionaries with potentially extra fields.

    Returns:
        List of messages with only 'role' and 'content' fields.
    """
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def build_evaluator_user_prompt(
    user_message: str,
    agent_reply: str,
    extended_history: List[Dict[str, Any]],
) -> str:
    """Construct the user prompt for the evaluator model.

    Args:
        user_message: The latest message from the user.
        agent_reply: The agent's response to evaluate.
        extended_history: Full conversation history including tool calls.

    Returns:
        Formatted prompt string for the evaluator.
    """
    user_prompt = f"Here's the conversation between the User and the Agent: \n\n{extended_history}\n\n"
    user_prompt += f"Here's the latest message from the User: \n\n{user_message}\n\n"
    user_prompt += f"Here's the latest response from the Agent: \n\n{agent_reply}\n\n"
    user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
    return user_prompt


def build_rerun_system_prompt(
    original_system_prompt: str,
    rejected_reply: str,
    feedback: str,
) -> str:
    """Build an updated system prompt for regeneration after rejection.

    Args:
        original_system_prompt: The original system prompt.
        rejected_reply: The response that was rejected.
        feedback: The evaluator's feedback explaining the rejection.

    Returns:
        Updated system prompt with rejection context.
    """
    updated_system_prompt = (
        original_system_prompt
        + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
    )
    updated_system_prompt += f"## Your attempted answer:\n{rejected_reply}\n\n"
    updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
    return updated_system_prompt


def convert_tools_to_openai_format(
    custom_tools: Optional[List[Union[Callable[..., Any], Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
    """Convert custom tool format to OpenAI format with auto-schema generation.

    Supports multiple input formats:
    1. Just function: [func1, func2] - Auto-generates everything from signature and docstring
    2. Dict with auto-gen: [{"function": func1}] - Auto-generates schema, optional description override
    3. Dict with manual: [{"function": func1, "parameters": {...}}] - Backward compatible

    Args:
        custom_tools: List of tools (functions or dicts with function references)

    Returns:
        tuple: (openai_tools, tool_map)
            - openai_tools: List of tools in OpenAI format
            - tool_map: Dict mapping function names to actual functions

    Raises:
        ValueError: If a dict tool is missing the 'function' key.
        TypeError: If a tool is not callable or dict, or if 'function' value is not callable.
    """
    if not custom_tools:
        return [], {}

    openai_tools: List[Dict[str, Any]] = []
    tool_map: Dict[str, Callable[..., Any]] = {}

    for tool in custom_tools:
        # Determine format and extract function
        if callable(tool):
            # Format 1: Just a function - auto-generate everything
            func: Callable[..., Any] = tool
            schema = extract_function_schema(func)
        elif isinstance(tool, dict):
            # Format 2 or 3: Dict with function
            func = tool.get("function")  # type: ignore

            if func is None:
                raise ValueError(f"Tool dict must have 'function' key. Got: {list(tool.keys())}")

            if not callable(func):
                raise TypeError(f"Tool 'function' must be callable, got {type(func).__name__}")

            if "parameters" in tool:
                # Format 3: Manual schema (backward compatible)
                schema = {
                    "name": func.__name__,
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters"],
                }
            else:
                # Format 2: Auto-generate schema from function
                schema = extract_function_schema(func)

                # Allow description override
                if "description" in tool:
                    schema["description"] = tool["description"]
        else:
            raise TypeError(f"Tool must be a callable or dict, got {type(tool).__name__}")

        func_name = func.__name__

        openai_tools.append({"type": "function", "function": schema})

        tool_map[func_name] = func

    return openai_tools, tool_map


def configure_logging(
    name: str = "chat_factory",
    level: str = "INFO",
    format: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> None:
    """
    Configure logging for the chat_factory library not for the MCP servers
    see ChatFactory.set_logging_level().

    This function provides a convenient way to configure logging for the library.
    It ensures a handler is configured and sets the log level.

    Note:
        For more control, users can configure logging directly using Python's
        logging module in their application code.

    Args:
        level: Log level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to "INFO".
        format: Optional custom format string for log messages.
                If not provided, uses a default format with timestamp and level.
        datefmt: Optional custom date format string.

    Examples:
    ::

        Basic usage - set log level to DEBUG:
        >>> from chat_factory.utils.factory_utils import configure_logging
        >>> configure_logging(level="DEBUG")

        Custom format:
        >>> configure_logging(
        ...     level="INFO",
        ...     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ...     datefmt="%Y-%m-%d %H:%M:%S"
        ... )

        Using standard logging module for more control:
        >>> import logging
        >>> logging.getLogger("chat_factory").setLevel(logging.DEBUG)
        >>> # Or configure entire app:
        >>> logging.basicConfig(level=logging.DEBUG)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure root logger has a handler configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # No handlers configured yet, set up basic configuration
        logging.basicConfig(
            level=log_level,
            format=format or "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        )

    # Set the log level
    library_logger = logging.getLogger(name)
    library_logger.setLevel(log_level)
