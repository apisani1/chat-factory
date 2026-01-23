"""Comprehensive unit tests for AsyncChatFactory."""

import asyncio
from contextlib import aclosing
from unittest.mock import (
    AsyncMock,
    MagicMock,
)

import pytest

from chat_factory import AsyncChatFactory
from chat_factory.utils.factory import Evaluation


# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


# ============================================================================
# Async Lifecycle Tests
# ============================================================================


class TestAsyncChatFactoryLifecycle:
    """Tests for AsyncChatFactory async lifecycle management."""

    async def test_async_context_manager(self, mock_async_chat_model):
        """Test async context manager usage."""
        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            assert factory is not None
            assert isinstance(factory, AsyncChatFactory)

    async def test_init_without_mcp(self, mock_async_chat_model):
        """Test initialization without MCP config."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        assert factory.generator_model == mock_async_chat_model
        assert factory.mcp_client is None
        assert factory.mcp_config_path is None

    async def test_async_connect_disconnect(self, mock_async_chat_model):
        """Test explicit connect/disconnect methods."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        # Connect (no MCP config, so no real connection)
        await factory.connect_to_mcp_servers()
        assert factory.mcp_client is None  # No config provided

        # Disconnect
        await factory.disconnect_from_mcp_servers()

    async def test_async_exit_cleanup(self, mock_async_chat_model):
        """Test AsyncExitStack cleanup on exit."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        async with factory:
            pass  # Exit context

        # _stack should be None after exit
        assert factory._stack is None

    async def test_async_no_mcp_works(self, mock_async_chat_model):
        """Test factory works without MCP configuration."""
        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            result = await factory.achat("Hello", [])

            assert result == "Async test response"

    async def test_multiple_enter_exit(self, mock_async_chat_model):
        """Test multiple context manager entries and exits."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        async with factory:
            pass

        # Can enter again
        async with factory:
            pass


# ============================================================================
# Async Initialization Tests
# ============================================================================


class TestAsyncChatFactoryInitialization:
    """Tests for AsyncChatFactory initialization."""

    async def test_init_with_tools(self, mock_async_chat_model, sample_tools):
        """Test AsyncChatFactory registers custom tools correctly."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model, tools=sample_tools)

        assert len(factory.openai_tools) == 2
        assert len(factory.tool_map) == 2
        assert "get_weather" in factory.tool_map

    async def test_init_with_evaluator(self, mock_async_chat_model, mock_async_evaluator_model):
        """Test AsyncChatFactory with evaluator model."""
        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=mock_async_evaluator_model,
            response_limit=3,
        )

        assert factory.evaluator_model == mock_async_evaluator_model
        assert factory.response_limit == 3

    async def test_init_with_kwargs(self, mock_async_chat_model):
        """Test AsyncChatFactory with kwargs."""
        gen_kwargs = {"temperature": 0.8}

        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            generator_kwargs=gen_kwargs,
        )

        assert factory.generator_kwargs == gen_kwargs


# ============================================================================
# Async Property Tests
# ============================================================================


class TestAsyncChatFactoryProperties:
    """Tests for AsyncChatFactory properties."""

    async def test_mcp_prompts_no_client(self, mock_async_chat_model):
        """Test mcp_prompts returns empty dict when no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        assert factory.mcp_prompts == {}

    async def test_prompt_names_no_client(self, mock_async_chat_model):
        """Test prompt_names returns empty list when no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        assert factory.prompt_names == []

    async def test_mcp_resources_no_client(self, mock_async_chat_model):
        """Test mcp_resources returns empty dict when no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        assert factory.mcp_resources == {}

    async def test_resource_names_no_client(self, mock_async_chat_model):
        """Test resource_names returns empty list when no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        assert factory.resource_names == []


# ============================================================================
# Concurrent Operation Tests
# ============================================================================


class TestAsyncChatFactoryConcurrency:
    """Tests for AsyncChatFactory concurrent operations."""

    async def test_concurrent_achat_calls(self, mock_async_chat_model):
        """Test multiple concurrent achat calls."""
        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            # Run multiple chats concurrently
            results = await asyncio.gather(
                factory.achat("Hello 1", []),
                factory.achat("Hello 2", []),
                factory.achat("Hello 3", []),
            )

            assert len(results) == 3
            for result in results:
                assert result == "Async test response"

    async def test_no_state_corruption(self, mock_async_chat_model):
        """Test concurrent calls don't corrupt shared state."""
        call_count = 0

        async def counting_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay to simulate async work
            return f"Response {call_count}"

        mock_async_chat_model.agenerate_response = counting_response

        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            results = await asyncio.gather(
                factory.achat("A", []),
                factory.achat("B", []),
            )

            assert len(results) == 2
            assert call_count == 2


# ============================================================================
# Async Streaming Tests
# ============================================================================


class TestAsyncChatFactoryStreaming:
    """Tests for AsyncChatFactory streaming functionality."""

    async def test_astream_chat_accumulate_true(self, mock_async_chat_model):
        """Test async streaming with accumulate=True."""

        async def mock_stream():
            for chunk in ["a", "b", "c"]:
                yield chunk

        mock_async_chat_model.astream_response = MagicMock(return_value=mock_stream())

        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            chunks = []
            async for chunk in factory.astream_chat("Hello", [], accumulate=True):
                chunks.append(chunk)

            assert chunks == ["a", "ab", "abc"]

    async def test_astream_chat_accumulate_false(self, mock_async_chat_model):
        """Test async streaming with accumulate=False (deltas)."""

        async def mock_stream():
            for chunk in ["x", "y", "z"]:
                yield chunk

        mock_async_chat_model.astream_response = MagicMock(return_value=mock_stream())

        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            chunks = []
            async for chunk in factory.astream_chat("Hello", [], accumulate=False):
                chunks.append(chunk)

            assert chunks == ["x", "y", "z"]

    async def test_astream_chat_exception(self, mock_async_chat_model):
        """Test async streaming handles exception gracefully."""

        async def failing_stream():
            yield "start"
            raise Exception("Stream error")

        mock_async_chat_model.astream_response = MagicMock(return_value=failing_stream())

        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            chunks = []
            async for chunk in factory.astream_chat("Hello", []):
                chunks.append(chunk)

            # Should have yielded start chunk, then error message
            assert len(chunks) >= 1
            assert "error" in chunks[-1].lower()

    async def test_astream_early_break(self, mock_async_chat_model):
        """Test breaking early from async stream."""

        async def long_stream():
            for i in range(100):
                yield str(i)

        mock_async_chat_model.astream_response = MagicMock(return_value=long_stream())

        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            chunks = []
            async with aclosing(factory.astream_chat("Hello", [], accumulate=False)) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
                    if len(chunks) >= 3:
                        break

            assert len(chunks) == 3

    async def test_get_async_stream_chat(self, mock_async_chat_model):
        """Test get_async_stream_chat returns the method."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        stream_fn = factory.get_async_stream_chat()

        assert stream_fn == factory.astream_chat
        assert callable(stream_fn)


# ============================================================================
# Async Tool Handling Tests
# ============================================================================


class TestAsyncChatFactoryToolHandling:
    """Tests for AsyncChatFactory tool handling."""

    async def test_async_handle_tool_call_custom(self, mock_async_chat_model, sample_tools):
        """Test handling custom tool call in async context."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model, tools=sample_tools)

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"city": "Paris"}'

        results = await factory.handle_tool_call([tool_call])

        assert len(results) == 1
        mock_async_chat_model.format_tool_result.assert_called_once()

    async def test_async_handle_tool_call_unknown(self, mock_async_chat_model):
        """Test handling unknown tool in async context."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "unknown"
        tool_call.function.arguments = "{}"

        results = await factory.handle_tool_call([tool_call])

        assert len(results) == 1
        mock_async_chat_model.format_tool_result.assert_called_with(tool_call_id="call_1", result={})

    async def test_async_tool_loop(self, mock_async_chat_model_with_tool_calls, sample_tools):
        """Test async tool calling loop."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model_with_tool_calls, tools=sample_tools)

        async with factory:
            result = await factory.achat("Calculate 5 + 3", [])

            assert result == "Async final response after tool"

    async def test_async_get_reply_with_tools(self, mock_async_chat_model_with_tool_calls, sample_tools):
        """Test async get_reply with tool calling."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model_with_tool_calls, tools=sample_tools)
        messages = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Calculate"}]

        reply, updated_messages = await factory.get_reply(messages)

        assert reply == "Async final response after tool"
        assert mock_async_chat_model_with_tool_calls.agenerate_response.call_count == 2

    async def test_async_get_reply_exception(self, mock_async_chat_model):
        """Test async get_reply handles exception."""
        mock_async_chat_model.agenerate_response = AsyncMock(side_effect=Exception("API Error"))

        factory = AsyncChatFactory(generator_model=mock_async_chat_model)
        messages = [{"role": "user", "content": "Hello"}]

        reply, _ = await factory.get_reply(messages)

        assert "error" in reply.lower()  # type: ignore

    async def test_async_handle_multiple_tool_calls(self, mock_async_chat_model, sample_tools):
        """Test handling multiple tool calls in async context."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model, tools=sample_tools)

        tool_call_1 = MagicMock()
        tool_call_1.id = "call_1"
        tool_call_1.function.name = "get_weather"
        tool_call_1.function.arguments = '{"city": "London"}'

        tool_call_2 = MagicMock()
        tool_call_2.id = "call_2"
        tool_call_2.function.name = "calculate"
        tool_call_2.function.arguments = '{"a": 1, "b": 2}'

        results = await factory.handle_tool_call([tool_call_1, tool_call_2])

        assert len(results) == 2


# ============================================================================
# Async Evaluator Tests
# ============================================================================


class TestAsyncChatFactoryEvaluator:
    """Tests for AsyncChatFactory evaluator pattern."""

    async def test_async_evaluate(self, mock_async_chat_model, mock_async_evaluator_model):
        """Test async evaluation."""
        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=mock_async_evaluator_model,
        )

        evaluation = await factory.evaluate("Hello", "Hi there!", [])

        assert evaluation.is_acceptable is True

    async def test_async_chat_with_evaluator(self, mock_async_chat_model, mock_async_evaluator_model):
        """Test async chat with evaluator."""
        async with AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=mock_async_evaluator_model,
        ) as factory:
            result = await factory.achat("Hello", [])

            assert result == "Async test response"
            mock_async_evaluator_model.agenerate_response.assert_called_once()

    async def test_async_evaluator_exception(self, mock_async_chat_model):
        """Test async evaluator returns acceptable on exception."""
        evaluator = MagicMock()
        evaluator.agenerate_response = AsyncMock(side_effect=Exception("Eval failed"))

        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=evaluator,
        )

        evaluation = await factory.evaluate("Hello", "Hi", [])

        assert evaluation.is_acceptable is True

    async def test_async_rerun(self, mock_async_chat_model):
        """Test async rerun with feedback."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        reply, messages = await factory.rerun(
            reply="Original",
            feedback="Make it better",
            extended_history=[
                {"role": "system", "content": "Prompt"},
                {"role": "user", "content": "Hello"},
            ],
        )

        # Verify system prompt was updated with feedback
        call_args = mock_async_chat_model.agenerate_response.call_args
        messages_arg = call_args[1]["messages"]
        assert "Original" in messages_arg[0]["content"]
        assert "Make it better" in messages_arg[0]["content"]

    async def test_async_evaluator_retries(self, mock_async_chat_model):
        """Test async evaluator retry loop."""
        evaluator = MagicMock()
        evaluator.agenerate_response = AsyncMock(
            side_effect=[
                Evaluation(is_acceptable=False, feedback="Too short"),
                Evaluation(is_acceptable=True, feedback="Good"),
            ]
        )

        mock_async_chat_model.agenerate_response = AsyncMock(side_effect=["First response", "Better response"])

        async with AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=evaluator,
            response_limit=5,
        ) as factory:
            result = await factory.achat("Hello", [])

            assert result == "Better response"
            assert evaluator.agenerate_response.call_count == 2

    async def test_async_max_retries(self, mock_async_chat_model):
        """Test async evaluator stops at response_limit."""
        evaluator = MagicMock()
        evaluator.agenerate_response = AsyncMock(return_value=Evaluation(is_acceptable=False, feedback="Not good"))

        mock_async_chat_model.agenerate_response = AsyncMock(return_value="Response")

        async with AsyncChatFactory(
            generator_model=mock_async_chat_model,
            evaluator_model=evaluator,
            response_limit=3,
        ) as factory:
            await factory.achat("Hello", [])

            # response_limit=3 means loop runs while responses < 3
            # So evaluator called 2 times (responses 1 and 2)
            assert evaluator.agenerate_response.call_count == 2


# ============================================================================
# Async MCP Integration Tests
# ============================================================================


class TestAsyncChatFactoryMCPIntegration:
    """Tests for AsyncChatFactory MCP integration."""

    async def test_async_set_mcp_logging_level_valid(self, mock_async_chat_model, mock_async_mcp_client):
        """Test async setting valid MCP logging level."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)
        factory.mcp_client = mock_async_mcp_client

        await factory.set_mcp_logging_level("INFO")

        mock_async_mcp_client.set_logging_level.assert_called_once_with(level="info")

    async def test_async_set_mcp_logging_level_invalid(self, mock_async_chat_model):
        """Test async invalid logging level raises ValueError."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        with pytest.raises(ValueError) as exc_info:
            await factory.set_mcp_logging_level("INVALID")

        assert "Invalid logging level" in str(exc_info.value)

    async def test_async_instantiate_prompt_no_client(self, mock_async_chat_model):
        """Test async instantiate_prompt with no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        result = await factory.instantiate_prompt("test", lambda x: {})

        assert result == []

    async def test_async_instantiate_resource_no_client(self, mock_async_chat_model):
        """Test async instantiate_resource with no MCP client."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        result = await factory.instantiate_resource("test", lambda x: {})

        assert result == []


# ============================================================================
# Async Edge Cases
# ============================================================================


class TestAsyncChatFactoryEdgeCases:
    """Tests for edge cases in async operations."""

    async def test_get_async_chat_returns_method(self, mock_async_chat_model):
        """Test get_async_chat returns the achat method."""
        factory = AsyncChatFactory(generator_model=mock_async_chat_model)

        chat_fn = factory.get_async_chat()

        assert chat_fn == factory.achat
        assert callable(chat_fn)

    async def test_empty_history(self, mock_async_chat_model, empty_history):
        """Test async chat with empty history."""
        async with AsyncChatFactory(generator_model=mock_async_chat_model) as factory:
            result = await factory.achat("Hello", empty_history)

            assert result == "Async test response"

    async def test_kwargs_passed_to_agenerate_response(self, mock_async_chat_model):
        """Test generator_kwargs are passed to agenerate_response."""
        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            generator_kwargs={"temperature": 0.9},
        )

        await factory.achat("Hello", [])

        call_kwargs = mock_async_chat_model.agenerate_response.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    async def test_display_content_callback(self, mock_async_chat_model):
        """Test display_content callback is stored."""
        callback = MagicMock()
        factory = AsyncChatFactory(
            generator_model=mock_async_chat_model,
            display_content=callback,
        )

        assert factory.display_content == callback
