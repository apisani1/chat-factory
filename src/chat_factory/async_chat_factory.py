import atexit
import json
from contextlib import AsyncExitStack
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

from dotenv import (
    find_dotenv,
    load_dotenv,
)

from .mcp_utils import process_tool_result_content
from .models import ChatModel
from .schema_utils import extract_function_schema


load_dotenv(find_dotenv(), override=True)


class Evaluation(BaseModel):
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


class AsyncChatFactory:
    """Factory for creating chat functions with optional MCP tool integration."""

    def __init__(
        self,
        generator_model: ChatModel,
        system_prompt: str = GENERATOR_PROMPT,
        evaluator_model: Optional[ChatModel] = None,
        evaluator_system_prompt: str = EVALUATOR_PROMPT,
        response_limit: int = 5,
        tools: Optional[List] = None,
        mcp_config_path: Optional[str] = None,
        *,
        generator_kwargs: Optional[Dict] = None,
        evaluator_kwargs: Optional[Dict] = None,
    ):
        """Initialize ChatFactory with models, tools, and optional MCP integration.

        Args:
            generator_model: Model for generating responses
            system_prompt: System prompt for generator
            evaluator_model: Optional model for evaluating responses
            evaluator_system_prompt: System prompt for evaluator
            response_limit: Max number of generation attempts
            tools: List of custom tools (functions or dicts)
            mcp_config_path: Optional path to MCP config file
            generator_kwargs: Additional kwargs for generator
            evaluator_kwargs: Additional kwargs for evaluator
        """
        # Store configuration
        self.generator_model = generator_model
        self.system_prompt = system_prompt
        self.evaluator_model = evaluator_model
        self.evaluator_system_prompt = evaluator_system_prompt
        self.response_limit = response_limit
        self.generator_kwargs = generator_kwargs or {}
        self.evaluator_kwargs = evaluator_kwargs or {}

        # Convert custom tools to OpenAI format
        self.openai_tools, self.tool_map = self._convert_tools_to_openai(tools)

        # Initialize MCP manager if config provided
        self.mcp_client: Optional[Any] = None
        self.mcp_config_path = mcp_config_path
        self._stack: Optional[AsyncExitStack] = None

        # Register shutdown handler
        atexit.register(lambda: print("Shutting down AsyncChatFactory..."))

    async def __aenter__(self) -> "AsyncChatFactory":
        """Enter the async context manager."""
        if self.mcp_config_path:
            try:
                from mcp_multi_server import MultiServerClient
                from mcp_multi_server.utils import (
                    mcp_tools_to_openai_format,
                    print_capabilities_summary,
                )

                # Create and initialize MCP client
                self.mcp_client = MultiServerClient.from_config(self.mcp_config_path)
                self._stack = AsyncExitStack()
                await self._stack.__aenter__()
                await self.mcp_client.connect_all(self._stack)
                print_capabilities_summary(self.mcp_client)

                # Get raw MCP tools and convert to OpenAI format
                mcp_tools = self.mcp_client.list_tools()
                mcp_tools_openai = mcp_tools_to_openai_format(mcp_tools.tools)
                self.openai_tools.extend(mcp_tools_openai)

            except ImportError as e:
                print(f"MCP Multi-Server package is not installed: {e}")
                print("Run pip install mcp-multi-server")
                self.mcp_client = None
            except Exception as e:
                print(f"Error initializing MCP client: {e}")
                self.mcp_client = None

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager."""
        if self._stack:
            print("Disconnecting from MCP servers...")
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
            self._stack = None
            print("MCP client closed successfully")

    async def connect_to_mcp_servers(self) -> "AsyncChatFactory":
        """Connect to MCP servers."""
        return await self.__aenter__()

    async def disconnect_from_mcp_servers(self) -> None:
        """Disconnect from MCP servers."""
        await self.__aexit__(None, None, None)

    @staticmethod
    def _convert_tools_to_openai(
        custom_tools: Optional[List[Union[Callable, Dict[str, Any]]]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
        """
        Convert custom tool format to OpenAI format with auto-schema generation.

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
        """
        if not custom_tools:
            return [], {}

        openai_tools = []
        tool_map = {}

        for tool in custom_tools:
            # Determine format and extract function
            if callable(tool):
                # Format 1: Just a function - auto-generate everything
                func: Callable = tool
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

    async def chat(self, message: str, history: List[Dict[str, Any]]) -> str:

        def evaluator_user_prompt(user_message: str, agent_reply: str, extended_history: List[Dict[str, Any]]) -> str:
            user_prompt = f"Here's the conversation between the User and the Agent: \n\n{extended_history}\n\n"
            user_prompt += f"Here's the latest message from the User: \n\n{user_message}\n\n"
            user_prompt += f"Here's the latest response from the Agent: \n\n{agent_reply}\n\n"
            user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
            return user_prompt

        async def evaluate(user_message: str, agent_reply: str, extended_history: List[Dict[str, Any]]) -> Evaluation:
            try:
                messages = [{"role": "system", "content": self.evaluator_system_prompt}] + [
                    {"role": "user", "content": evaluator_user_prompt(user_message, agent_reply, extended_history)}
                ]
                evaluation = self.evaluator_model.generate_response(  # type: ignore
                    messages=messages, response_format=Evaluation, **self.evaluator_kwargs
                )
                assert isinstance(evaluation, Evaluation)
                return evaluation
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return Evaluation(is_acceptable=True, feedback="")

        def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Remove extra fields from messages that may be added by Gradio or other UIs."""
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        async def handle_tool_call(tool_calls: List[Any]) -> List[Dict[str, Any]]:
            """Handle tool calls - uses self.tool_map and self.mcp_client."""
            results = []
            for tool_call in tool_calls:

                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"Tool called: {tool_name}", flush=True)

                tool = self.tool_map.get(tool_name)
                if tool:
                    # Custom tool function
                    result = tool(**arguments)
                elif self.mcp_client:
                    # MCP tool
                    mcp_tool_result = await self.mcp_client.call_tool(tool_name, arguments)
                    result = process_tool_result_content(mcp_tool_result)
                else:
                    # Unknown tool
                    result = {}

                # print(f"Tool result: {result}", flush=True)
                results.append(self.generator_model.format_tool_result(tool_call_id=tool_call.id, result=result))
            return results

        async def get_reply(
            extended_history: List[Dict[str, Any]],
        ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
            """Generate reply with tool calling support."""
            messages = extended_history.copy()
            try:
                reply = self.generator_model.generate_response(
                    messages=messages,
                    tools=self.openai_tools,
                    **self.generator_kwargs,
                )
                while isinstance(reply, list):
                    messages.append({"role": "assistant", "content": None, "tool_calls": reply})
                    messages += await handle_tool_call(reply)
                    reply = self.generator_model.generate_response(
                        messages=messages,
                        tools=self.openai_tools,
                        **self.generator_kwargs,
                    )
                return reply, messages
            except Exception as e:
                print(f"Error generating reply: {e}")
                return "Sorry, I encountered an error while generating a response.", messages

        async def rerun(
            reply: str, feedback: str, extended_history: List[Dict[str, Any]]
        ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
            """Regenerate reply based on evaluator feedback."""
            updated_system_prompt = (
                self.system_prompt
                + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
            )
            updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
            updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
            messages = [{"role": "system", "content": updated_system_prompt}] + extended_history[
                1:
            ]  # exclude previous system prompt
            return await get_reply(messages)

        # Main chat logic
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )
        reply, extended_history = await get_reply(messages)

        if self.evaluator_model:
            responses = 1
            while responses < self.response_limit:

                evaluation = await evaluate(message, reply, extended_history)  # type: ignore

                if evaluation.is_acceptable:
                    print("Passed evaluation - returning reply")
                    break

                print("Failed evaluation - retrying")
                print(evaluation.feedback)
                reply, extended_history = await rerun(reply, evaluation.feedback, extended_history)  # type: ignore
                responses += 1

            print(f"****Final response after {responses} attempt(s).")

        return reply  # type: ignore

    def get_async_gradio_chat(self) -> Callable[[str, List[Dict[str, Any]]], Coroutine[Any, Any, str]]:
        return self.chat
