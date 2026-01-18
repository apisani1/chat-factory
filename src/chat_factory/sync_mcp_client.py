"""
Synchronous wrapper for MCP MultiServerClient with background event loop.

This module provides SyncMultiServerClient, a context manager that wraps
the async MultiServerClient from mcp_multi_server in a synchronous interface
using a background thread with persistent event loop.
"""

import asyncio
import atexit
import logging
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Union,
)

from pydantic import AnyUrl

from mcp.types import (
    CallToolResult,
    EmptyResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    LoggingLevel,
    ReadResourceResult,
    TextContent,
)
from mcp_multi_server import MultiServerClient


logger = logging.getLogger(__name__)


class SyncMultiServerClient:
    """Manages MCP multi server client in a background thread with persistent event loop.

    This class provides a synchronous interface to the async MultiServerClient,
    making it easier to integrate with synchronous code while maintaining
    the efficiency of async operations.

    Usage:
        # Context manager (recommended)
        with SyncMultiServerClient(config_path) as client:
            tools = client.list_tools()
            result = client.call_tool("tool_name", {"arg": "value"})

        # Manual lifecycle management
        client = SyncMultiServerClient(config_path)
        tools = client.list_tools()
        # ... use client ...
        client.shutdown()

    Thread Safety:
        All public methods are thread-safe, using asyncio.run_coroutine_threadsafe()
        to schedule operations on the background event loop.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SyncMultiServerClient.

        Starts background thread and initializes MCP client during construction.
        Automatically registers cleanup handler to ensure proper shutdown on program exit.

        Args:
            config_path: Path to MCP configuration file.
            config_dict: Configuration dictionary (alternative to config_path).

        Raises:
            ValueError: If neither or both config_path and config_dict are provided.
            Exception: If MCP client initialization fails.
        """
        if (config_path is None) == (config_dict is None):
            raise ValueError("Exactly one of config_path or config_dict must be provided")

        self.config_path = config_path
        self.config_dict = config_dict
        self.mcp_client: Optional[MultiServerClient] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._init_complete = threading.Event()  # For cross-thread signaling
        self._loop_ready = threading.Event()  # Signal when event loop is ready
        self._lifecycle_future = None  # Will hold the lifecycle task future

        # Start background thread
        self._start_background_loop()

        # Start long-running lifecycle management task
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(
            self._manage_client_lifecycle(), self.loop  # type: ignore
        )

        # Wait for initialization to complete (blocks until MCP client is ready)
        if not self._init_complete.wait(timeout=30):
            raise RuntimeError("MCP client initialization timed out after 30 seconds")

        # Check if lifecycle task failed during initialization
        # (future is done = error occurred, future still running = success)
        if self._lifecycle_future.done():
            exc = self._lifecycle_future.exception()
            if exc:
                raise exc

        # Register automatic cleanup on program exit
        # This ensures MCP client is properly shutdown when the program terminates,
        atexit.register(self.shutdown)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "SyncMultiServerClient":
        """Create a client from a configuration file path.

        This is a convenience class method that's equivalent to calling the constructor
        with a config_path argument.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            A new SyncMultiServerClient instance.

        Examples:
            >>> client = SyncMultiServerClient.from_config("mcp_config.json")
        """
        return cls(config_path=config_path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SyncMultiServerClient":
        """Create a client from a configuration dictionary.

        This method allows programmatic configuration without needing a JSON file.

        Args:
            config_dict: Dictionary containing server configurations in the same
                format as the JSON file (with "mcpServers" key).

        Returns:
            A new SyncMultiServerClient instance with the provided configuration.

        Examples:
            >>> config = {
            ...     "mcpServers": {
            ...         "tool_server": {
            ...             "command": "python",
            ...             "args": ["-m", "my_package.tool_server"]
            ...         }
            ...     }
            ... }
            >>> client = SyncMultiServerClient.from_dict(config)
        """
        return cls(config_dict=config_dict)

    def _start_background_loop(self) -> None:
        """Start background thread with event loop."""
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True, name="MCPClientThread")
        self.thread.start()

        # Wait for loop to be ready (blocks efficiently, no CPU burn)
        self._loop_ready.wait()

    def _run_event_loop(self) -> None:
        """Run persistent event loop in background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._loop_ready.set()  # Signal that loop is ready
        self.loop.run_forever()

    async def _manage_client_lifecycle(self) -> None:
        """Long-running task that manages the entire MCP client context lifecycle.

        This ensures __aenter__ and __aexit__ happen in the same async task,
        preventing "cancel scope in different task" errors.

        Flow:
            1. Enter MCP client async context (__aenter__)
            2. Signal initialization complete
            3. Stay alive until shutdown requested
            4. Exit context (__aexit__) in the SAME task
        """
        try:
            # Enter the MCP client context (creates cancel scope in THIS task)
            if self.config_path is not None:
                self.mcp_client = MultiServerClient.from_config(self.config_path)
            else:
                self.mcp_client = MultiServerClient.from_dict(self.config_dict)  # type: ignore[arg-type]
            await self.mcp_client.__aenter__()

            # Signal that initialization is complete
            self._init_complete.set()

            # Stay alive until shutdown is requested
            # This keeps the cancel scope alive in this task
            while not self._shutdown:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("Error in MCP client lifecycle: %s", e)
            self._init_complete.set()  # Unblock __init__ even on error
            raise

        finally:
            # Exit the context in the SAME task (no cancel scope error!)
            if self.mcp_client is not None:
                try:
                    await self.mcp_client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error("Error closing MCP client: %s", e)

    def shutdown(self) -> None:
        """Shutdown background thread and cleanup MCP client.

        Safe to call multiple times. Waits up to 10 seconds for graceful cleanup.
        """
        logger.debug("Shutting down SyncMultiServerClient...")
        if self.loop is not None and not self._shutdown:
            self._shutdown = True

            # Deadlock prevention: if called from event loop thread,
            # we can't block waiting on the lifecycle future
            if threading.current_thread() is self.thread:
                self.loop.call_soon(self.loop.stop)
                return

            try:
                # Signal shutdown and wait for lifecycle task to complete
                # The task will exit the MCP client context properly
                if self._lifecycle_future is not None:
                    self._lifecycle_future.result(timeout=10)
                    logger.debug("MCP client closed successfully")
            except Exception as e:
                # Errors expected during interpreter shutdown
                logger.warning("Error during MCP client shutdown: %s", e)

            try:
                # Stop event loop
                self.loop.call_soon_threadsafe(self.loop.stop)

                # Wait for thread to finish
                if self.thread is not None:
                    self.thread.join(timeout=5)
            except Exception:
                # Thread might already be stopped during interpreter shutdown
                pass

    def __enter__(self) -> "SyncMultiServerClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Exit context manager and cleanup."""
        self.shutdown()
        return False  # Don't suppress exceptions

    def set_logging_level(self, level: LoggingLevel) -> EmptyResult:
        """Set the logging level for the multi-server client and the MCP connected servers.

        Args:
            level: Logging level as a string in lower case (e.g., "debug", "info", "notice", "warning", "error",
                "critical", "alert", "emergency") as defined in MCP LoggingLevel.

        Note:
            The following mappings of MCP to Python logging leves are applied:
            - "notice" -> "WARNING"
            - "alert" and "emergency" -> "CRITICAL"

        Examples:
            >>> SyncMultiServerClient.set_logging_level("debug")
        """
        if level not in {"debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"}:
            raise ValueError(
                f""""
                Invalid logging level: {level}.
                See: https://modelcontextprotocol.github.io/python-sdk/api/#mcp.ClientSession.set_logging_level")
            """
            )
        if level == "notice":
            level = "warning"
        elif level in ("alert", "emergency"):
            level = "critical"

        if self.loop is None or self.mcp_client is None:
            raise RuntimeError("MCP client not initialized")

        # Schedule async call on background event loop
        future = asyncio.run_coroutine_threadsafe(self._set_logging_level_async(level), self.loop)
        future.result()  # Wait for completion
        return EmptyResult()

    async def _set_logging_level_async(self, level: LoggingLevel) -> None:
        """Async implementation of set_logging_level (runs in background loop)."""
        if self.mcp_client is None:
            raise ValueError("MCP client not initialized")
        await self.mcp_client.set_logging_level(level=level)

    def list_tools(self) -> ListToolsResult:
        """List available MCP tools in raw MCP format.

        Returns:
            List of MCP Tool objects (not converted to OpenAI format).
            Returns empty list if client not initialized or error occurs.
        """
        if self.mcp_client is None:
            return ListToolsResult(tools=[])

        try:
            return self.mcp_client.list_tools()
        except Exception as e:
            logger.error("Error listing MCP tools: %s", e)
            return ListToolsResult(tools=[])

    def list_prompts(self) -> ListPromptsResult:
        """Get combined list of all prompts from all servers.

        Returns:
            ListPromptsResult containing all prompts from all servers with
            server attribution in the serverName meta field.
            Returns empty list if client not initialized or error occurs.
        """
        if self.mcp_client is None:
            return ListPromptsResult(prompts=[], nextCursor=None)

        try:
            return self.mcp_client.list_prompts()
        except Exception as e:
            logger.error("Error listing prompts: %s", e)
            return ListPromptsResult(prompts=[], nextCursor=None)

    def list_resources(self, use_namespace: bool = True) -> ListResourcesResult:
        """Get combined list of all resources from all servers.

        Args:
            use_namespace: Whether to namespace URIs with server name (default True).
                When True, URIs are in format "server_name:original_uri" for auto-routing.

        Returns:
            ListResourcesResult containing all resources from all servers with
            server attribution in the serverName meta field.
            Returns empty list if client not initialized or error occurs.
        """
        if self.mcp_client is None:
            return ListResourcesResult(resources=[], nextCursor=None)

        try:
            return self.mcp_client.list_resources(use_namespace=use_namespace)
        except Exception as e:
            logger.error("Error listing resources: %s", e)
            return ListResourcesResult(resources=[], nextCursor=None)

    def list_resource_templates(self, use_namespace: bool = True) -> ListResourceTemplatesResult:
        """Get combined list of all resource templates from all servers.

        Args:
            use_namespace: Whether to namespace URI templates with server name (default True).
                When True, templates are in format "server_name:original_template".

        Returns:
            ListResourceTemplatesResult containing all templates from all servers with
            server attribution in the serverName meta field.
            Returns empty list if client not initialized or error occurs.
        """
        if self.mcp_client is None:
            return ListResourceTemplatesResult(resourceTemplates=[], nextCursor=None)

        try:
            return self.mcp_client.list_resource_templates(use_namespace=use_namespace)
        except Exception as e:
            logger.error("Error listing resource templates: %s", e)
            return ListResourceTemplatesResult(resourceTemplates=[], nextCursor=None)

    def _create_error_result(self, error_message: str) -> CallToolResult:
        """Create a CallToolResult indicating an error.

        Args:
            error_message: The error message to include in the result.

        Returns:
            CallToolResult with isError=True and the error message in content.
        """
        return CallToolResult(
            content=[TextContent(type="text", text=error_message)],
            isError=True,
        )

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> CallToolResult:
        """Call MCP tool synchronously with optional timeout.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dictionary
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            CallToolResult object. If timeout occurs, returns an error result.
        """
        if self.loop is None or self.mcp_client is None:
            return self._create_error_result("MCP client not initialized")

        future = asyncio.run_coroutine_threadsafe(self._call_tool_async(tool_name, arguments, **kwargs), self.loop)

        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            return self._create_error_result(f"MCP tool '{tool_name}' timed out after {timeout} seconds")

    async def _call_tool_async(self, tool_name: str, arguments: Dict, **kwargs: Any) -> CallToolResult:
        """Async implementation of tool call (runs in background loop).

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Processed tool result as CallToolResult object
        """
        try:
            if self.mcp_client is None:
                raise ValueError("MCP client not initialized")

            return await self.mcp_client.call_tool(tool_name, arguments, **kwargs)
        except Exception as e:
            logger.error("Error calling MCP tool '%s': %s", tool_name, e)
            return self._create_error_result(f"Error calling MCP tool '{tool_name}': {e}")

    def read_resource(
        self,
        uri: Union[str, AnyUrl],
        server_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ReadResourceResult:
        """Read a resource with optional auto-routing via namespaced URIs.

        Args:
            uri: Resource URI. Can be namespaced as "server:uri" for auto-routing.
                URIs from list_resources() are already namespaced for convenience.
            server_name: Optional explicit server name. If provided, assumes that
                there is no namespace in the provided URI.
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            ReadResourceResult containing the resource content.
            Returns empty result if client not initialized or timeout occurs.

        Examples:
            >>> # Auto-routing with namespaced URI (from list_resources())
            >>> resources = client.list_resources().resources
            >>> result = client.read_resource(resources[0].uri)

            >>> # Explicit server (no namespace in URI)
            >>> result = client.read_resource("file:///path", server_name="filesystem")

            >>> # Manual namespacing
            >>> result = client.read_resource("filesystem:file:///path")
        """
        if self.loop is None or self.mcp_client is None:
            return ReadResourceResult(contents=[])

        future = asyncio.run_coroutine_threadsafe(self._read_resource_async(uri, server_name), self.loop)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error("Read resource timed out after %s seconds", timeout)
            return ReadResourceResult(contents=[])

    async def _read_resource_async(self, uri: Union[str, AnyUrl], server_name: Optional[str]) -> ReadResourceResult:
        """Async implementation of read_resource (runs in background loop)."""
        if self.mcp_client is None:
            raise ValueError("MCP client not initialized")
        return await self.mcp_client.read_resource(uri, server_name=server_name)

    def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
        server_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> GetPromptResult:
        """Get a prompt by automatically routing to the appropriate server.

        Args:
            name: Name of the prompt to get.
            arguments: Optional arguments for the prompt.
            server_name: Optional server name to explicitly specify which server to use.
                If not provided, the server will be automatically determined from
                the prompt name.
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            GetPromptResult containing the prompt messages.
            Returns empty result if client not initialized or timeout occurs.

        Examples:
            >>> # Auto-routing (prompt name determines server)
            >>> result = client.get_prompt("greeting", arguments={"name": "World"})

            >>> # Explicit server
            >>> result = client.get_prompt("greeting", server_name="prompts_server")
        """
        if self.loop is None or self.mcp_client is None:
            return GetPromptResult(messages=[])

        future = asyncio.run_coroutine_threadsafe(self._get_prompt_async(name, arguments, server_name), self.loop)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error("Get prompt '%s' timed out after %s seconds", name, timeout)
            return GetPromptResult(messages=[])

    async def _get_prompt_async(
        self,
        name: str,
        arguments: Optional[Dict[str, str]],
        server_name: Optional[str],
    ) -> GetPromptResult:
        """Async implementation of get_prompt (runs in background loop)."""
        if self.mcp_client is None:
            raise ValueError("MCP client not initialized")
        return await self.mcp_client.get_prompt(name, arguments=arguments, server_name=server_name)
