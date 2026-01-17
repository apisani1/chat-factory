"""Synchronous LLM client supporting multiple providers."""

import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from pydantic import BaseModel

from anthropic import Anthropic
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from .utils.model_utils import (
    OPENAI_CLIENT_MAP,
    convert_tool_calls_to_openai,
    convert_tools_to_anthropic,
    format_tool_result,
    prepare_messages_for_anthropic,
    prepare_tool_params,
)


class ChatModel:
    """Synchronous LLM client supporting multiple providers."""

    def __init__(
        self, model_name: str, provider: str = "openai", api_key: Optional[str] = None, **kwargs: Any
    ) -> None:
        self.client: Union[OpenAI, Anthropic]
        self.model_name = model_name
        self._provider = provider
        if provider in OPENAI_CLIENT_MAP:
            api_key = api_key or os.getenv(OPENAI_CLIENT_MAP[provider]["env_var"])
            if not api_key:
                raise ValueError(
                    f"Missing API key for {provider} and {OPENAI_CLIENT_MAP[provider]['env_var']} not found in the environment either."
                )
            self.client = OpenAI(base_url=OPENAI_CLIENT_MAP[provider]["base_url"], api_key=api_key, **kwargs)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return self._provider

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 10000,
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel, List[ChatCompletionMessageToolCall]]:
        """
        Generate a response from the LLM using the configured provider.

        Supports text responses, structured responses, and tool calling across multiple
        LLM providers (OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Standard roles: 'system', 'user', 'assistant', 'tool'.
            max_tokens: Maximum tokens in the response (default: 10000).
                       Only used for Anthropic; OpenAI uses default from API.
            response_format: Pydantic model class for structured responses.
                        Cannot be used with 'tools' parameter.
            tools: List of tool definitions for function calling (default: None).
                  Use OpenAI format: [{"type": "function", "function": {...}}]
                  Automatically converted for Anthropic.
                  Cannot be used with 'response_format' parameter.
            **kwargs: Additional provider-specific parameters passed to the API.

        Returns:
            str | BaseModel | list: Return type depends on parameters:
                - Text mode: Returns str
                - Structured response mode: Returns Pydantic model instance
                - Tool calling mode: Returns list of tool calls in OpenAI format
                  (caller handles tool execution and looping)

        Raises:
            ValueError: If both tools and response_format are provided.

        Examples:
            Basic text response:
                >>> model = ChatModel("gpt-4o-mini")
                >>> response = model.generate_response([
                ...     {"role": "user", "content": "Hello!"}
                ... ])

            Structured response:
                >>> class Answer(BaseModel):
                ...     text: str
                ...     confidence: float
                >>> response = model.generate_response(
                ...     messages=[{"role": "user", "content": "What is 2+2?"}],
                ...     response_format=Answer
                ... )

            Tool calling (caller handles loop):
                >>> tools = [{
                ...     "type": "function",
                ...     "function": {
                ...         "name": "get_weather",
                ...         "description": "Get weather for a location",
                ...         "parameters": {
                ...             "type": "object",
                ...             "properties": {
                ...                 "location": {"type": "string"}
                ...             },
                ...             "required": ["location"]
                ...         }
                ...     }
                ... }]
                >>> messages = [{"role": "user", "content": "What's the weather in SF?"}]
                >>> response = model.generate_response(messages, tools=tools)
                >>> # Response is either str or list of tool calls (if tools requested)
                >>> if isinstance(response, list):  # Tool calls returned
                ...     for tool_call in response:
                ...         # Execute tool using tool_call["function"]["name"]
                ...         # and tool_call["function"]["arguments"]
                ...         result = execute_tool(tool_call)
                ...         # Format and add to messages
                ...         messages.append(model.format_tool_result(
                ...             tool_call["id"], result
                ...         ))
                ...     # Call again with tool results
                ...     final_response = model.generate_response(messages, tools=tools)

        Note:
            - OpenAI: Uses native parse API for structured responses
            - Anthropic: Uses tool calling for structured responses
            - System messages: Automatically handled per provider requirements
            - Tool format: Use OpenAI format; automatically converted for Anthropic
            - Tool results: Use format_tool_result() helper to format tool results
        """

        if response_format is not None and tools is not None:
            raise ValueError(
                "Cannot use both 'tools' and 'response_format' parameters together. "
                "Use 'tools' for function calling or 'response_format' for structured output, not both."
            )

        if isinstance(self.client, OpenAI):
            if response_format is not None:
                # Structured response mode via native parse API
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name, messages=messages, response_format=response_format, **kwargs  # type: ignore
                )
                return response.choices[0].message.parsed  # type: ignore

            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, tools=tools, **kwargs  # type: ignore
            )

            if response.choices[0].finish_reason == "tool_calls":
                # Tool calling mode
                return response.choices[0].message.tool_calls  # type: ignore

            # Regular text response mode
            return response.choices[0].message.content  # type: ignore

        if isinstance(self.client, Anthropic):
            # Anthropic API differences:
            # 1. Uses separate 'system' parameter instead of system messages in the array
            # 2. Uses tool calling (function calling) for structured output instead of a native parse API
            system_content, anthropic_messages = prepare_messages_for_anthropic(messages)

            if response_format is not None:
                # Prepare request for structured response
                tool_params = prepare_tool_params(response_format)

            else:
                # Prepare regular tool calling parameters
                tool_params = convert_tools_to_anthropic(tools)

            response = self.client.messages.create(  # type: ignore
                model=self.model_name,
                messages=anthropic_messages,  # type: ignore
                max_tokens=max_tokens,
                system=system_content,
                **tool_params,
                **kwargs,
            )

            tool_calls = [block for block in response.content if block.type == "tool_use"]  # type: ignore
            if tool_calls:
                if response_format is not None:
                    # Structured response mode - return parsed model instance
                    tool_use = tool_calls[0]
                    return response_format(**tool_use.input)

                # Tool calling mode
                return convert_tool_calls_to_openai(tool_calls)

            # Regular text response mode
            return response.content[0].text  # type: ignore

        raise ValueError(f"Unsupported client type: {type(self.client).__name__}")

    def stream_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 10000,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream text chunks from the LLM.

        Yields text chunks as they arrive from the provider. Does not support
        tool calling or structured output - use generate_response() for those.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            max_tokens: Maximum tokens in the response (default: 10000).
                       Only used for Anthropic; OpenAI uses default from API.
            **kwargs: Additional provider-specific parameters passed to the API.

        Yields:
            str: Text chunks as they arrive from the LLM.

        Examples:
            >>> model = ChatModel("gpt-4o-mini")
            >>> for chunk in model.stream_response([{"role": "user", "content": "Hello!"}]):
            ...     print(chunk, end="", flush=True)
        """
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                stream=True,
                **kwargs,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:  # type: ignore[union-attr]
                    yield chunk.choices[0].delta.content  # type: ignore[union-attr]

        elif isinstance(self.client, Anthropic):
            system_content, anthropic_messages = prepare_messages_for_anthropic(messages)

            with self.client.messages.stream(
                model=self.model_name,
                messages=anthropic_messages,  # type: ignore
                max_tokens=max_tokens,
                system=system_content,
                **kwargs,
            ) as stream:
                yield from stream.text_stream

        else:
            raise ValueError(f"Unsupported client type: {type(self.client).__name__}")

    def format_tool_result(self, tool_call_id: str, result: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Format tool execution result for adding back to conversation.

        Handles provider-specific formatting differences between OpenAI and Anthropic.

        Args:
            tool_call_id: ID of the tool call from the response
            result: Result from tool execution (dict or str)

        Returns:
            dict: Formatted message to append to conversation history

        Examples:
            >>> result_msg = model.format_tool_result(
            ...     tool_call_id="call_123",
            ...     result={"temp": 72, "condition": "sunny"}
            ... )
            >>> messages.append(result_msg)
        """
        return format_tool_result(self._provider, tool_call_id, result)
