import atexit
import json
import logging
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
from mcp_multi_server.utils import substitute_template_variables

from .async_models import AsyncChatModel
from .utils.factory import (
    EVALUATOR_PROMPT,
    GENERATOR_PROMPT,
    Evaluation,
    build_evaluator_user_prompt,
    build_rerun_system_prompt,
    convert_tools_to_openai_format,
    sanitize_messages,
)
from .utils.mcp import (
    convert_mcp_content_to_message,
    process_tool_result_content,
    search_prompt,
    search_resource,
)


logger = logging.getLogger(__name__)

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
        display_content: Optional[Callable] = None,
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
        self.display_content = display_content
        self.response_limit = response_limit
        self.generator_kwargs = generator_kwargs or {}
        self.evaluator_kwargs = evaluator_kwargs or {}

        # Convert custom tools to OpenAI format
        self.openai_tools, self.tool_map = convert_tools_to_openai_format(tools)

        # Initialize MCP manager if config provided
        self.mcp_client: Optional[Any] = None
        self.mcp_config_path = mcp_config_path
        self._stack: Optional[AsyncExitStack] = None
        self._mcp_prompts: Dict[str, Any] = {}
        self._prompt_names: List[str] = []
        self._mcp_resources: Dict[str, Any] = {}
        self._resource_names: List[str] = []
        self._mcp_resource_templates: Dict[str, Any] = {}
        self._template_names: List[str] = []

        # Register shutdown handler
        atexit.register(lambda: logger.debug("Shutting down AsyncChatFactory..."))

    async def __aenter__(self) -> "AsyncChatFactory":
        """Enter the async context manager."""
        if self.mcp_config_path:
            try:
                from mcp_multi_server import MultiServerClient
                from mcp_multi_server.utils import mcp_tools_to_openai_format

                # Create and initialize MCP client
                self.mcp_client = MultiServerClient.from_config(self.mcp_config_path)

                self.mcp_client.mcp_log_level = "CRITICAL"

                self._stack = AsyncExitStack()
                await self._stack.__aenter__()
                await self.mcp_client.connect_all(self._stack)

                # Get raw MCP tools and convert to OpenAI format
                mcp_tools = self.mcp_client.list_tools()
                mcp_tools_openai = mcp_tools_to_openai_format(mcp_tools.tools)
                self.openai_tools.extend(mcp_tools_openai)

                # Get MCP prompts, resources and resource templates
                mcp_prompts = self.mcp_client.list_prompts().prompts
                self._mcp_prompts = {prompt.name: prompt for prompt in mcp_prompts}
                self._prompt_names = [prompt.name for prompt in mcp_prompts]
                mcp_resources = self.mcp_client.list_resources().resources
                self._mcp_resources = {resource.name: resource for resource in mcp_resources}
                self._resource_names = [resource.name for resource in mcp_resources]
                mcp_resource_templates = self.mcp_client.list_resource_templates().resourceTemplates
                self._mcp_resource_templates = {template.name: template for template in mcp_resource_templates}
                self._template_names = [template.name for template in mcp_resource_templates]

            except ImportError as e:
                logger.error("MCP Multi-Server package is not installed: %s. Run: pip install mcp-multi-server", e)
                self.mcp_client = None
            except Exception as e:
                logger.error("Error initializing MCP client: %s", e)
                self.mcp_client = None

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager."""
        if self._stack:
            logger.debug("Disconnecting from MCP servers...")
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
            self._stack = None
            logger.debug("MCP client closed successfully")

    async def connect_to_mcp_servers(self) -> "AsyncChatFactory":
        """Connect to MCP servers."""
        return await self.__aenter__()

    async def disconnect_from_mcp_servers(self) -> None:
        """Disconnect from MCP servers."""
        await self.__aexit__(None, None, None)

    @property
    def mcp_prompts(self) -> Dict[str, Any]:
        """Get MCP prompts if MCP client is initialized."""
        if self.mcp_client:
            return self._mcp_prompts
        return {}

    @property
    def prompt_names(self) -> List[str]:
        """Get MCP prompt names if MCP client is initialized."""
        if self.mcp_client:
            return self._prompt_names
        return []

    @property
    def mcp_resources(self) -> Dict[str, Any]:
        """Get MCP resources if MCP client is initialized."""
        if self.mcp_client:
            return {**self._mcp_resources, **self._mcp_resource_templates}
        return {}

    @property
    def resource_names(self) -> List[str]:
        """Get MCP resource names if MCP client is initialized."""
        if self.mcp_client:
            return self._resource_names + self._template_names
        return []

    async def set_mcp_logging_level(self, level: str) -> None:
        """Set the logging level the MCP connected servers.

        Args:
            level: Logging level as string (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
        """
        log_level = level.upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid logging level: {level}. Choose from DEBUG, INFO, WARNING, ERROR, CRITICAL.")
        if self.mcp_client:
            try:
                await self.mcp_client.set_logging_level(level=log_level.lower())
                logger.info("MCP logging level set to %s", log_level)
            except Exception as e:
                logger.warning("Error setting MCP logging level to %s: %s", log_level, e)

    async def instantiate_prompt(
        self, prompt_name: str, get_prompt_arguments: Callable, display_content: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve a MCP prompt by name and convert to OpenAI message format.
        Args:
            prompt_name: Name of the MCP prompt to retrieve.
            get_prompt_arguments: Function to get prompt arguments through UI.

        Returns:
            List of OpenAI-formatted messages with proper image/audio support.
        """

        if not self.mcp_client:
            return []
        prompt, prompt_arguments = search_prompt(self.mcp_prompts, prompt_name)
        if not prompt:
            return []
        arguments = get_prompt_arguments(prompt_arguments)

        prompt_result = await self.mcp_client.get_prompt(prompt_name, arguments=arguments)

        openai_messages = []
        for msg in prompt_result.messages:
            if display_content:
                display_content(msg.content)
            content = convert_mcp_content_to_message(msg.content)
            openai_messages.append({"role": msg.role, "content": content})

        return openai_messages

    async def instantiate_resource(
        self, resource_name: str, get_template_variables: Callable, display_result: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve a resource by name from the list of resources.
        Args:
            resource_name: Name of the resource to retrieve.
            get_template_variables: Function to get template variable values through UI.
            is_template: Whether the resource is a template.
        Returns:
            List of OpenAI-formatted messages with resource content.
        """
        if not self.mcp_client:
            return []
        resource, uri, variables = search_resource(self.mcp_resources, resource_name)
        if not resource:
            return []
        if variables:
            var_values = get_template_variables(variables)
            uri = substitute_template_variables(uri, var_values)  # type: ignore

        resource_result = await self.mcp_client.read_resource(uri=uri)

        # Assuming single text message resource
        resource_result_text = (
            resource_result.contents[0].text  # type: ignore
            if resource_result.contents and hasattr(resource_result.contents[0], "text")
            else ""
        )
        if display_result:
            display_result(resource_result)
        return [{"role": "user", "content": resource_result_text}]

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
            logger.error("Error during evaluation: %s", e)
            return Evaluation(is_acceptable=True, feedback="")

    async def handle_tool_call(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
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
                mcp_tool_result = await self.mcp_client.call_tool(tool_name, arguments)
                result = process_tool_result_content(mcp_tool_result, self.display_content)
            else:
                # Unknown tool
                result = {}

            logger.debug("Tool result: %s", result)
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
            logger.error("Error generating reply: %s", e)
            return "Sorry, I encountered an error while generating a response.", messages

    async def rerun(
        self, reply: str, feedback: str, extended_history: List[Dict[str, Any]]
    ) -> Tuple[Union[str, BaseModel], List[Dict[str, Any]]]:
        """Regenerate reply based on evaluator feedback."""
        updated_system_prompt = build_rerun_system_prompt(self.system_prompt, reply, feedback)
        messages = [{"role": "system", "content": updated_system_prompt}] + extended_history[1:]
        return await self.get_reply(messages)

    async def achat(self, message: str, history: List[Dict[str, Any]]) -> str:
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
        reply, extended_history = await self.get_reply(messages)

        if self.evaluator_model:
            responses = 1
            while responses < self.response_limit:

                evaluation = await self.evaluate(message, reply, extended_history)  # type: ignore

                if evaluation.is_acceptable:
                    logger.info("Passed evaluation - returning reply")
                    break

                logger.info("Failed evaluation - retrying")
                logger.info(evaluation.feedback)
                reply, extended_history = await self.rerun(reply, evaluation.feedback, extended_history)  # type: ignore
                responses += 1

            logger.info("****Final response after %d attempt(s).", responses)

        return reply  # type: ignore

    def get_async_chat(self) -> Callable[[str, List[Dict[str, Any]]], Coroutine[Any, Any, str]]:
        return self.achat

    async def astream_chat(
        self, message: str, history: List[Dict[str, Any]], *, accumulate: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Async stream chat response.

        Note: Tool calling and evaluator are not supported in streaming mode.
        Use achat() method for tool calling support.

        Args:
            message: User message to respond to
            history: Conversation history
            accumulate: If True, yield accumulated text (for Gradio).
                       If False, yield individual chunks/deltas.

        Yields:
            str: Response text (accumulated or delta based on accumulate parameter)
        """
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )

        try:
            if accumulate:
                accumulated = ""
                async for chunk in self.generator_model.astream_response(
                    messages=messages,
                    **self.generator_kwargs,
                ):
                    accumulated += chunk
                    yield accumulated
            else:
                async for chunk in self.generator_model.astream_response(
                    messages=messages,
                    **self.generator_kwargs,
                ):
                    yield chunk
        except Exception as e:
            logger.error("Error during streaming: %s", e)
            yield f"Sorry, I encountered an error: {e}"

    def get_async_stream_chat(
        self,
    ) -> Callable[[str, List[Dict[str, Any]]], AsyncGenerator[str, None]]:
        """Return async streaming chat function for Gradio."""
        return self.astream_chat
