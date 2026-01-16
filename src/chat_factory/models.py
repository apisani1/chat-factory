import json
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

from anthropic import (
    Anthropic,
    AsyncAnthropic,
)
from openai import (
    AsyncOpenAI,
    OpenAI,
)
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


class ChatModel:
    def __init__(
        self, model_name: str, provider: str = "openai", api_key: Optional[str] = None, **kwargs: Any
    ) -> None:
        self.client: Union[OpenAI, Anthropic]
        self._async_client: Optional[Union[AsyncOpenAI, AsyncAnthropic]] = None
        self.model_name = model_name
        self._provider = provider
        self._kwargs = kwargs
        self._api_key: Optional[str] = None
        self._base_url: Optional[str] = None
        if provider in OPENAI_CLIENT_MAP:
            api_key = api_key or os.getenv(OPENAI_CLIENT_MAP[provider]["env_var"])
            if not api_key:
                raise ValueError(
                    f"Missing API key for {provider} and {OPENAI_CLIENT_MAP[provider]['env_var']} not found in the environment either."
                )
            self._api_key = api_key
            self._base_url = OPENAI_CLIENT_MAP[provider]["base_url"]
            self.client = OpenAI(base_url=self._base_url, api_key=api_key, **kwargs)
        elif provider == "anthropic":
            self._api_key = api_key
            self._base_url = None
            self.client = Anthropic(api_key=api_key, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    @property
    def async_client(self) -> Union[AsyncOpenAI, AsyncAnthropic]:
        """Lazily create and return the async client."""
        if self._async_client is None:
            if self._provider in OPENAI_CLIENT_MAP:
                self._async_client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key, **self._kwargs)
            elif self._provider == "anthropic":
                self._async_client = AsyncAnthropic(api_key=self._api_key, **self._kwargs)
        return self._async_client  # type: ignore

    @staticmethod
    def _prepare_messages_for_anthropic(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
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

    @staticmethod
    def _prepare_tool_params(response_format: Optional[type[BaseModel]]) -> Dict[str, Any]:
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

    @staticmethod
    def _convert_tools_to_anthropic(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
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

    @staticmethod
    def _convert_tool_calls_to_openai(tool_use_blocks: List[Any]) -> List[ChatCompletionMessageToolCall]:
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
            system_content, anthropic_messages = self._prepare_messages_for_anthropic(messages)

            if response_format is not None:
                # Prepare request for structured response
                tool_params = self._prepare_tool_params(response_format)

            else:
                # Prepare regular tool calling parameters
                tool_params = self._convert_tools_to_anthropic(tools)

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
                return self._convert_tool_calls_to_openai(tool_calls)

            # Regular text response mode
            return response.content[0].text  # type: ignore

        raise ValueError(f"Unsupported client type: {type(self.client).__name__}")

    async def agenerate_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 10000,
        response_format: Optional[type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel, List[ChatCompletionMessageToolCall]]:
        """
        Async version of generate_response.

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
        """

        if response_format is not None and tools is not None:
            raise ValueError(
                "Cannot use both 'tools' and 'response_format' parameters together. "
                "Use 'tools' for function calling or 'response_format' for structured output, not both."
            )

        client = self.async_client

        if isinstance(client, AsyncOpenAI):
            if response_format is not None:
                # Structured response mode via native parse API
                response = await client.beta.chat.completions.parse(
                    model=self.model_name, messages=messages, response_format=response_format, **kwargs  # type: ignore
                )
                return response.choices[0].message.parsed  # type: ignore

            response = await client.chat.completions.create(
                model=self.model_name, messages=messages, tools=tools, **kwargs  # type: ignore
            )

            if response.choices[0].finish_reason == "tool_calls":
                # Tool calling mode
                return response.choices[0].message.tool_calls  # type: ignore

            # Regular text response mode
            return response.choices[0].message.content  # type: ignore

        if isinstance(client, AsyncAnthropic):
            # Anthropic API differences:
            # 1. Uses separate 'system' parameter instead of system messages in the array
            # 2. Uses tool calling (function calling) for structured output instead of a native parse API
            system_content, anthropic_messages = self._prepare_messages_for_anthropic(messages)

            if response_format is not None:
                # Prepare request for structured response
                tool_params = self._prepare_tool_params(response_format)

            else:
                # Prepare regular tool calling parameters
                tool_params = self._convert_tools_to_anthropic(tools)

            response = await client.messages.create(
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
                return self._convert_tool_calls_to_openai(tool_calls)

            # Regular text response mode
            return response.content[0].text  # type: ignore

        raise ValueError(f"Unsupported client type: {type(client).__name__}")

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
            system_content, anthropic_messages = self._prepare_messages_for_anthropic(messages)

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

    async def astream_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 10000,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async stream text chunks from the LLM.

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
            >>> async for chunk in model.astream_response([{"role": "user", "content": "Hello!"}]):
            ...     print(chunk, end="", flush=True)
        """
        client = self.async_client

        if isinstance(client, AsyncOpenAI):
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                stream=True,
                **kwargs,
            )
            async for chunk in response:  # type: ignore[union-attr]
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif isinstance(client, AsyncAnthropic):
            system_content, anthropic_messages = self._prepare_messages_for_anthropic(messages)

            async with client.messages.stream(
                model=self.model_name,
                messages=anthropic_messages,  # type: ignore
                max_tokens=max_tokens,
                system=system_content,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        else:
            raise ValueError(f"Unsupported client type: {type(client).__name__}")

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
        if isinstance(self.client, OpenAI):
            return {
                "role": "tool",
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
                "tool_call_id": tool_call_id,
            }
        if isinstance(self.client, Anthropic):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    }
                ],
            }
        raise ValueError(f"Unsupported client type: {type(self.client).__name__}")
