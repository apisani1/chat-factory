"""
Synchronous wrapper for MCP MultiServerClient with background event loop.

This module provides SyncMultiServerClient, a context manager that wraps
the async MultiServerClient from mcp_multi_server in a synchronous interface
using a background thread with persistent event loop.
"""

import asyncio
import atexit
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
)

from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
)
from mcp_multi_server import MultiServerClient


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

    def __init__(self, config_path: str):
        """Initialize SyncMultiServerClient with config path.

        Starts background thread and initializes MCP client during construction.
        Automatically registers cleanup handler to ensure proper shutdown on program exit.

        Args:
            config_path: Path to MCP configuration file

        Raises:
            Exception: If MCP client initialization fails
        """
        self.config_path = config_path
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
            self.mcp_client = MultiServerClient.from_config(self.config_path)
            await self.mcp_client.__aenter__()

            # Signal that initialization is complete
            self._init_complete.set()

            # Stay alive until shutdown is requested
            # This keeps the cancel scope alive in this task
            while not self._shutdown:
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in MCP client lifecycle: {e}")
            self._init_complete.set()  # Unblock __init__ even on error
            raise

        finally:
            # Exit the context in the SAME task (no cancel scope error!)
            if self.mcp_client is not None:
                try:
                    await self.mcp_client.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Error closing MCP client: {e}")

    def list_tools(self) -> ListToolsResult:
        """List available MCP tools in raw MCP format.

        Returns:
            List of MCP Tool objects (not converted to OpenAI format).
            Returns empty list if client not initialized or error occurs.
        """
        if self.mcp_client is None:
            return ListToolsResult(tools=[])

        try:
            # Access list_tools() synchronously - it's not async
            return self.mcp_client.list_tools()
        except Exception as e:
            print(f"Error listing MCP tools: {e}")
            return ListToolsResult(tools=[])

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
            print(f"Error calling MCP tool '{tool_name}': {e}")
            return self._create_error_result(f"Error calling MCP tool '{tool_name}': {e}")

    def shutdown(self) -> None:
        """Shutdown background thread and cleanup MCP client.

        Safe to call multiple times. Waits up to 10 seconds for graceful cleanup.
        """
        print("Shutting down SyncMultiServerClient...")
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
                    print("MCP client closed successfully")
            except Exception as e:
                # Errors expected during interpreter shutdown
                print(f"Error during MCP client shutdown: {e}")

            try:
                # Stop event loop
                self.loop.call_soon_threadsafe(self.loop.stop)

                # Wait for thread to finish
                if self.thread is not None:
                    self.thread.join(timeout=5)
            except Exception:
                # Thread might already be stopped during interpreter shutdown
                pass

    # Context manager support
    def __enter__(self) -> "SyncMultiServerClient":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Exit context manager and cleanup."""
        self.shutdown()
        return False  # Don't suppress exceptions
