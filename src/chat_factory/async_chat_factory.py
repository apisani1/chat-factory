import atexit
import json
from contextlib import AsyncExitStack
from typing import (
    Any,
    AsyncGenerator,
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

from .async_models import AsyncChatModel
from .utils.factory_utils import (
    EVALUATOR_PROMPT,
    GENERATOR_PROMPT,
    Evaluation,
    build_evaluator_user_prompt,
    build_rerun_system_prompt,
    convert_tools_to_openai_format,
    sanitize_messages,
)
from .utils.mcp_utils import process_tool_result_content


load_dotenv(find_dotenv(), override=True)


class AsyncChatFactory:
    """Factory for creating chat functions with optional MCP tool integration."""

    def __init__(
        self,
        generator_model: AsyncChatModel,
        system_prompt: str = GENERATOR_PROMPT,
        evaluator_model: Optional[AsyncChatModel] = None,
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
        self.openai_tools, self.tool_map = convert_tools_to_openai_format(tools)

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

    async def evaluate(
        self, user_message: str, agent_reply: str, extended_history: List[Dict[str, Any]]
    ) -> Evaluation:
        """Evaluate the agent's response using the evaluator model."""
        try:
            messages = [{"role": "system", "content": self.evaluator_system_prompt}] + [
                {"role": "user", "content": build_evaluator_user_prompt(user_message, agent_reply, extended_history)}
            ]
            evaluation = await self.evaluator_model.agenerate_response(  # type: ignore
                messages=messages, response_format=Evaluation, **self.evaluator_kwargs
            )
            assert isinstance(evaluation, Evaluation)
            return evaluation
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return Evaluation(is_acceptable=True, feedback="")

    async def handle_tool_call(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
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
        self,
        extended_history: List[Dict[str, Any]],
    ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
        """Generate reply with tool calling support."""
        messages = extended_history.copy()
        try:
            reply = await self.generator_model.agenerate_response(
                messages=messages,
                tools=self.openai_tools,
                **self.generator_kwargs,
            )
            while isinstance(reply, list):
                messages.append({"role": "assistant", "content": None, "tool_calls": reply})
                messages += await self.handle_tool_call(reply)
                reply = await self.generator_model.agenerate_response(
                    messages=messages,
                    tools=self.openai_tools,
                    **self.generator_kwargs,
                )
            return reply, messages
        except Exception as e:
            print(f"Error generating reply: {e}")
            return "Sorry, I encountered an error while generating a response.", messages

    async def rerun(
        self, reply: str, feedback: str, extended_history: List[Dict[str, Any]]
    ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
        """Regenerate reply based on evaluator feedback."""
        updated_system_prompt = build_rerun_system_prompt(self.system_prompt, reply, feedback)
        messages = [{"role": "system", "content": updated_system_prompt}] + extended_history[1:]
        return await self.get_reply(messages)

    async def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )
        reply, extended_history = await self.get_reply(messages)

        if self.evaluator_model:
            responses = 1
            while responses < self.response_limit:

                evaluation = await self.evaluate(message, reply, extended_history)  # type: ignore

                if evaluation.is_acceptable:
                    print("Passed evaluation - returning reply")
                    break

                print("Failed evaluation - retrying")
                print(evaluation.feedback)
                reply, extended_history = await self.rerun(reply, evaluation.feedback, extended_history)  # type: ignore
                responses += 1

            print(f"****Final response after {responses} attempt(s).")

        return reply  # type: ignore

    def get_async_gradio_chat(self) -> Callable[[str, List[Dict[str, Any]]], Coroutine[Any, Any, str]]:
        return self.chat

    async def astream_chat(self, message: str, history: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """
        Async stream chat response with tool calling support (hybrid mode).

        Handles tool calls non-streaming first, then streams the final text response.
        Note: Evaluator is not supported in streaming mode.

        Args:
            message: User message to respond to
            history: Conversation history

        Yields:
            str: Accumulated response text (Gradio expects accumulated, not deltas)
        """
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )

        # Phase 1: Handle tool calls (non-streaming)
        if self.openai_tools:
            try:
                reply = await self.generator_model.agenerate_response(
                    messages=messages,
                    tools=self.openai_tools,
                    **self.generator_kwargs,
                )

                # Tool calling loop
                while isinstance(reply, list):
                    messages.append({"role": "assistant", "content": None, "tool_calls": reply})
                    messages += await self.handle_tool_call(reply)
                    reply = await self.generator_model.agenerate_response(
                        messages=messages,
                        tools=self.openai_tools,
                        **self.generator_kwargs,
                    )

                # If we got a string from tool loop, simulate streaming
                if isinstance(reply, str):
                    accumulated = ""
                    for char in reply:
                        accumulated += char
                        yield accumulated
                    return

            except Exception as e:
                print(f"Error during tool handling: {e}")
                yield f"Sorry, I encountered an error: {e}"
                return

        # Phase 2: Stream final response (no tools or tools already handled)
        try:
            accumulated = ""
            async for chunk in self.generator_model.astream_response(
                messages=messages,
                **self.generator_kwargs,
            ):
                accumulated += chunk
                yield accumulated
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"Sorry, I encountered an error: {e}"

    def get_async_gradio_stream_chat(
        self,
    ) -> Callable[[str, List[Dict[str, Any]]], AsyncGenerator[str, None]]:
        """Return async streaming chat function for Gradio."""
        return self.astream_chat
