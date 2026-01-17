import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
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

from .models import ChatModel
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


logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)


class ChatFactory:
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
        self.openai_tools, self.tool_map = convert_tools_to_openai_format(tools)

        # Initialize MCP manager if config provided
        self.mcp_client = None
        if mcp_config_path:
            try:
                from mcp_multi_server.utils import mcp_tools_to_openai_format

                from .sync_mcp_client import SyncMultiServerClient

                # Create and initialize MCP client
                self.mcp_client = SyncMultiServerClient(mcp_config_path)

                # Get raw MCP tools and convert to OpenAI format
                mcp_tools = self.mcp_client.list_tools()
                mcp_tools_openai = mcp_tools_to_openai_format(mcp_tools.tools)
                self.openai_tools.extend(mcp_tools_openai)

            except ImportError as e:
                logger.error("MCP Multi-Server package is not installed: %s. Run: pip install mcp-multi-server", e)
                self.mcp_client = None
            except Exception as e:
                logger.error("Error initializing MCP client: %s", e)
                self.mcp_client = None

    def set_mcp_logging_level(self, level: str) -> None:
        """Set the logging level for the MCP connected servers.

        Args:
            level: Logging level as string (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
        """
        log_level = level.upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid logging level: {level}. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        if self.mcp_client:
            try:
                self.mcp_client.set_logging_level(level=log_level.lower())  # type: ignore
                logger.info("MCP logging level set to %s", log_level)
            except Exception as e:
                logger.warning("Error setting MCP logging level to %s: %s", log_level, e)

    def evaluate(self, user_message: str, agent_reply: str, extended_history: List[Dict[str, Any]]) -> Evaluation:
        """Evaluate the agent's response using the evaluator model."""
        try:
            messages = [{"role": "system", "content": self.evaluator_system_prompt}] + [
                {"role": "user", "content": build_evaluator_user_prompt(user_message, agent_reply, extended_history)}
            ]
            evaluation = self.evaluator_model.generate_response(  # type: ignore
                messages=messages, response_format=Evaluation, **self.evaluator_kwargs
            )
            assert isinstance(evaluation, Evaluation)
            return evaluation
        except Exception as e:
            logger.error("Error during evaluation: %s", e)
            return Evaluation(is_acceptable=True, feedback="")

    def handle_tool_call(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Handle tool calls - uses self.tool_map and self.mcp_client."""
        results = []
        for tool_call in tool_calls:

            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info("Tool called: %s", tool_name)

            tool = self.tool_map.get(tool_name)
            if tool:
                # Custom tool function
                result = tool(**arguments)
            elif self.mcp_client:
                # MCP tool
                mcp_tool_result = self.mcp_client.call_tool(tool_name, arguments)
                result = process_tool_result_content(mcp_tool_result)
            else:
                # Unknown tool
                result = {}

            logger.debug("Tool result: %s", result)
            results.append(self.generator_model.format_tool_result(tool_call_id=tool_call.id, result=result))
        return results

    def get_reply(self, extended_history: List[Dict[str, Any]]) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
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
                messages += self.handle_tool_call(reply)
                reply = self.generator_model.generate_response(
                    messages=messages,
                    tools=self.openai_tools,
                    **self.generator_kwargs,
                )
            return reply, messages
        except Exception as e:
            logger.error("Error generating reply: %s", e)
            return "Sorry, I encountered an error while generating a response.", messages

    def rerun(
        self, reply: str, feedback: str, extended_history: List[Dict[str, Any]]
    ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
        """Regenerate reply based on evaluator feedback."""
        updated_system_prompt = build_rerun_system_prompt(self.system_prompt, reply, feedback)
        messages = [{"role": "system", "content": updated_system_prompt}] + extended_history[1:]
        return self.get_reply(messages)

    def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """Process a chat message and return a response.

        Handles the complete chat flow including tool calling and optional
        evaluation with retry logic when an evaluator model is configured.

        Args:
            message: The user's message to respond to.
            history: Conversation history as a list of message dicts with
                'role' and 'content' keys.

        Returns:
            The assistant's response string.
        """
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )
        reply, extended_history = self.get_reply(messages)

        if self.evaluator_model:
            responses = 1
            while responses < self.response_limit:

                evaluation = self.evaluate(message, reply, extended_history)  # type: ignore

                if evaluation.is_acceptable:
                    logger.info("Passed evaluation - returning reply")
                    break

                logger.info("Failed evaluation - retrying")
                logger.info(evaluation.feedback)
                reply, extended_history = self.rerun(reply, evaluation.feedback, extended_history)  # type: ignore
                responses += 1

            logger.info("****Final response after %d attempt(s).", responses)

        return reply  # type: ignore

    def get_gradio_chat(self) -> Callable[[str, List[Dict[str, Any]]], str]:
        return self.chat

    def stream_chat(self, message: str, history: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Stream chat response.

        Note: Tool calling and evaluator are not supported in streaming mode.
        Use chat() method for tool calling support.

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

        try:
            accumulated = ""
            for chunk in self.generator_model.stream_response(
                messages=messages,
                **self.generator_kwargs,
            ):
                accumulated += chunk
                yield accumulated
        except Exception as e:
            logger.error("Error during streaming: %s", e)
            yield f"Sorry, I encountered an error: {e}"

    def get_gradio_stream_chat(self) -> Callable[[str, List[Dict[str, Any]]], Generator[str, None, None]]:
        """Return streaming chat function for Gradio."""
        return self.stream_chat
