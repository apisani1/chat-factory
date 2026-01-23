"""Comprehensive unit tests for ChatFactory."""

from unittest.mock import MagicMock

import pytest

from chat_factory import ChatFactory
from chat_factory.utils.factory import Evaluation


# ============================================================================
# Initialization Tests
# ============================================================================


class TestChatFactoryInitialization:
    """Tests for ChatFactory initialization."""

    def test_init_minimal(self, mock_chat_model):
        """Test ChatFactory creation with minimal arguments."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.generator_model == mock_chat_model
        assert factory.evaluator_model is None
        assert factory.mcp_client is None
        assert factory.openai_tools == []
        assert factory.tool_map == {}

    def test_init_with_system_prompt(self, mock_chat_model):
        """Test ChatFactory with custom system prompt."""
        custom_prompt = "You are a helpful assistant."
        factory = ChatFactory(generator_model=mock_chat_model, system_prompt=custom_prompt)

        assert factory.system_prompt == custom_prompt

    def test_init_with_tools(self, mock_chat_model, sample_tools):
        """Test ChatFactory registers custom tools correctly."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=sample_tools)

        assert len(factory.openai_tools) == 2
        assert len(factory.tool_map) == 2
        assert "get_weather" in factory.tool_map
        assert "calculate" in factory.tool_map

    def test_init_with_tool_dict(self, mock_chat_model, sample_tool_dict):
        """Test ChatFactory with dict-format tool."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=[sample_tool_dict])

        assert len(factory.openai_tools) == 1
        assert "my_tool" in factory.tool_map

    def test_init_with_evaluator(self, mock_chat_model, mock_evaluator_model):
        """Test ChatFactory with evaluator model."""
        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=mock_evaluator_model,
            response_limit=3,
        )

        assert factory.evaluator_model == mock_evaluator_model
        assert factory.response_limit == 3

    def test_init_with_kwargs(self, mock_chat_model):
        """Test ChatFactory with generator and evaluator kwargs."""
        gen_kwargs = {"temperature": 0.7}
        eval_kwargs = {"max_tokens": 100}

        factory = ChatFactory(
            generator_model=mock_chat_model,
            generator_kwargs=gen_kwargs,
            evaluator_kwargs=eval_kwargs,
        )

        assert factory.generator_kwargs == gen_kwargs
        assert factory.evaluator_kwargs == eval_kwargs

    def test_init_kwargs_default_to_empty_dict(self, mock_chat_model):
        """Test that kwargs default to empty dicts when not provided."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.generator_kwargs == {}
        assert factory.evaluator_kwargs == {}

    def test_init_mcp_config_not_found(self, mock_chat_model):
        """Test ChatFactory handles missing MCP config file gracefully."""
        # When config file doesn't exist, SyncMultiServerClient is still created
        # but will error in its lifecycle thread. The factory should not crash.
        factory = ChatFactory(
            generator_model=mock_chat_model,
            mcp_config_path="nonexistent_config.json",
        )

        # Should not crash during initialization
        # Note: The mcp_client may be created but will fail during lifecycle
        # The important thing is that the factory is usable
        assert factory is not None
        factory.shutdown()  # Clean up


# ============================================================================
# Context Manager Tests
# ============================================================================


class TestChatFactoryContextManager:
    """Tests for ChatFactory context manager behavior."""

    def test_context_manager_enter_exit(self, mock_chat_model):
        """Test basic context manager usage."""
        with ChatFactory(generator_model=mock_chat_model) as factory:
            assert factory is not None
            assert isinstance(factory, ChatFactory)

    def test_context_manager_calls_shutdown(self, mock_chat_model, mock_mcp_client):
        """Test that exiting context manager calls shutdown."""
        factory = ChatFactory(generator_model=mock_chat_model)
        factory.mcp_client = mock_mcp_client

        with factory:
            pass  # Exit context

        mock_mcp_client.shutdown.assert_called_once()
        assert factory.mcp_client is None

    def test_context_manager_exception_cleanup(self, mock_chat_model, mock_mcp_client):
        """Test that MCP client is cleaned up even on exception."""
        factory = ChatFactory(generator_model=mock_chat_model)
        factory.mcp_client = mock_mcp_client

        with pytest.raises(ValueError):
            with factory:
                raise ValueError("Test exception")

        mock_mcp_client.shutdown.assert_called_once()

    def test_shutdown_idempotent(self, mock_chat_model):
        """Test that shutdown can be called multiple times safely."""
        factory = ChatFactory(generator_model=mock_chat_model)

        # Should not raise
        factory.shutdown()
        factory.shutdown()
        factory.shutdown()


# ============================================================================
# Property Tests
# ============================================================================


class TestChatFactoryProperties:
    """Tests for ChatFactory properties."""

    def test_mcp_prompts_no_client(self, mock_chat_model):
        """Test mcp_prompts returns empty dict when no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.mcp_prompts == {}

    def test_prompt_names_no_client(self, mock_chat_model):
        """Test prompt_names returns empty list when no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.prompt_names == []

    def test_mcp_resources_no_client(self, mock_chat_model):
        """Test mcp_resources returns empty dict when no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.mcp_resources == {}

    def test_resource_names_no_client(self, mock_chat_model):
        """Test resource_names returns empty list when no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        assert factory.resource_names == []


# ============================================================================
# Tool Handling Tests
# ============================================================================


class TestChatFactoryToolHandling:
    """Tests for ChatFactory tool handling."""

    def test_handle_tool_call_custom_tool(self, mock_chat_model, sample_tools):
        """Test handling custom tool call."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=sample_tools)

        # Create mock tool call
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"city": "London"}'

        results = factory.handle_tool_call([tool_call])

        assert len(results) == 1
        mock_chat_model.format_tool_result.assert_called_once()

    def test_handle_tool_call_unknown_tool(self, mock_chat_model):
        """Test handling unknown tool returns empty dict."""
        factory = ChatFactory(generator_model=mock_chat_model)

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "unknown_tool"
        tool_call.function.arguments = "{}"

        results = factory.handle_tool_call([tool_call])

        assert len(results) == 1
        # format_tool_result called with empty dict result
        mock_chat_model.format_tool_result.assert_called_with(tool_call_id="call_1", result={})

    def test_handle_tool_call_multiple_tools(self, mock_chat_model, sample_tools):
        """Test handling multiple tool calls in single request."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=sample_tools)

        tool_call_1 = MagicMock()
        tool_call_1.id = "call_1"
        tool_call_1.function.name = "get_weather"
        tool_call_1.function.arguments = '{"city": "London"}'

        tool_call_2 = MagicMock()
        tool_call_2.id = "call_2"
        tool_call_2.function.name = "calculate"
        tool_call_2.function.arguments = '{"a": 5, "b": 3}'

        results = factory.handle_tool_call([tool_call_1, tool_call_2])

        assert len(results) == 2
        assert mock_chat_model.format_tool_result.call_count == 2

    def test_handle_tool_call_custom_priority_over_mcp(self, mock_chat_model, sample_tools, mock_mcp_client):
        """Test custom tools take priority over MCP tools with same name."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=sample_tools)
        factory.mcp_client = mock_mcp_client

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_weather"  # Same name as custom tool
        tool_call.function.arguments = '{"city": "London"}'

        factory.handle_tool_call([tool_call])

        # MCP client should NOT be called - custom tool takes priority
        mock_mcp_client.call_tool.assert_not_called()

    def test_handle_tool_call_mcp_tool(self, mock_chat_model, mock_mcp_client):
        """Test handling MCP tool call when no custom tool matches."""
        factory = ChatFactory(generator_model=mock_chat_model)
        factory.mcp_client = mock_mcp_client

        # Mock MCP tool result
        mcp_result = MagicMock()
        mcp_result.content = []
        mock_mcp_client.call_tool.return_value = mcp_result

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "mcp_tool"
        tool_call.function.arguments = '{"arg": "value"}'

        factory.handle_tool_call([tool_call])

        mock_mcp_client.call_tool.assert_called_once_with("mcp_tool", {"arg": "value"})

    def test_tool_returns_dict(self, mock_chat_model):
        """Test tool that returns dict is handled correctly."""

        def dict_tool() -> dict:
            """Returns a dict.

            Args: None
            """
            return {"key": "value", "number": 42}

        factory = ChatFactory(generator_model=mock_chat_model, tools=[dict_tool])

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "dict_tool"
        tool_call.function.arguments = "{}"

        factory.handle_tool_call([tool_call])

        # Verify format_tool_result was called with dict result
        call_args = mock_chat_model.format_tool_result.call_args
        assert call_args[1]["result"] == {"key": "value", "number": 42}

    def test_tool_returns_string(self, mock_chat_model):
        """Test tool that returns string is handled correctly."""

        def string_tool() -> str:
            """Returns a string.

            Args: None
            """
            return "Hello, World!"

        factory = ChatFactory(generator_model=mock_chat_model, tools=[string_tool])

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "string_tool"
        tool_call.function.arguments = "{}"

        factory.handle_tool_call([tool_call])

        call_args = mock_chat_model.format_tool_result.call_args
        assert call_args[1]["result"] == "Hello, World!"


# ============================================================================
# Response Generation Tests
# ============================================================================


class TestChatFactoryResponseGeneration:
    """Tests for ChatFactory response generation."""

    def test_get_reply_simple(self, mock_chat_model):
        """Test get_reply with no tool calls."""
        factory = ChatFactory(generator_model=mock_chat_model)
        messages = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Hello"}]

        reply, updated_messages = factory.get_reply(messages)

        assert reply == "Test response"
        mock_chat_model.generate_response.assert_called_once()

    def test_get_reply_with_tool_loop(self, mock_chat_model_with_tool_calls, sample_tools):
        """Test get_reply handles tool calling loop."""
        factory = ChatFactory(generator_model=mock_chat_model_with_tool_calls, tools=sample_tools)
        messages = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Weather in London?"}]

        reply, updated_messages = factory.get_reply(messages)

        assert reply == "Final response after tool"
        # Should be called twice: once for tool call, once for final response
        assert mock_chat_model_with_tool_calls.generate_response.call_count == 2

    def test_get_reply_exception_returns_error_message(self, mock_chat_model):
        """Test get_reply returns error message on exception."""
        mock_chat_model.generate_response.side_effect = Exception("API Error")
        factory = ChatFactory(generator_model=mock_chat_model)
        messages = [{"role": "user", "content": "Hello"}]

        reply, _ = factory.get_reply(messages)

        assert "error" in reply.lower()  # type: ignore

    def test_chat_basic(self, mock_chat_model, sample_history):
        """Test basic chat method."""
        factory = ChatFactory(generator_model=mock_chat_model)

        result = factory.chat("How are you?", sample_history)

        assert result == "Test response"

    def test_chat_empty_history(self, mock_chat_model, empty_history):
        """Test chat with empty history."""
        factory = ChatFactory(generator_model=mock_chat_model)

        result = factory.chat("Hello", empty_history)

        assert result == "Test response"

    def test_chat_sanitizes_history(self, mock_chat_model):
        """Test chat removes extra fields from history."""
        factory = ChatFactory(generator_model=mock_chat_model)

        # History with extra Gradio fields
        history_with_extras = [
            {"role": "user", "content": "Hi", "extra_field": "should_be_removed"},
            {"role": "assistant", "content": "Hello", "metadata": {"timestamp": 123}},
        ]

        factory.chat("Test", history_with_extras)

        # Verify generate_response was called with sanitized messages
        call_args = mock_chat_model.generate_response.call_args
        messages = call_args[1]["messages"]

        # Check that extra fields are not in the sanitized messages
        for msg in messages:
            if msg["role"] != "system":
                assert "extra_field" not in msg
                assert "metadata" not in msg

    def test_chat_with_tools(self, mock_chat_model_with_tool_calls, sample_tools):
        """Test chat with tool calling flow."""
        factory = ChatFactory(generator_model=mock_chat_model_with_tool_calls, tools=sample_tools)

        result = factory.chat("What's the weather?", [])

        assert result == "Final response after tool"

    def test_get_chat_returns_method(self, mock_chat_model):
        """Test get_chat returns the chat method."""
        factory = ChatFactory(generator_model=mock_chat_model)

        chat_fn = factory.get_chat()

        assert chat_fn == factory.chat
        assert callable(chat_fn)


# ============================================================================
# Evaluator Pattern Tests
# ============================================================================


class TestChatFactoryEvaluator:
    """Tests for ChatFactory evaluator pattern."""

    def test_evaluate_acceptable(self, mock_chat_model, mock_evaluator_model):
        """Test evaluate returns acceptable evaluation."""
        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=mock_evaluator_model,
        )

        evaluation = factory.evaluate("Hello", "Hi there!", [])

        assert evaluation.is_acceptable is True
        assert evaluation.feedback == ""

    def test_evaluate_rejected(self, mock_chat_model):
        """Test evaluate returns rejection with feedback."""
        evaluator = MagicMock()
        evaluator.generate_response.return_value = Evaluation(
            is_acceptable=False,
            feedback="Response is too short",
        )

        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=evaluator,
        )

        evaluation = factory.evaluate("Hello", "Hi", [])

        assert evaluation.is_acceptable is False
        assert "too short" in evaluation.feedback

    def test_evaluate_exception_returns_acceptable(self, mock_chat_model):
        """Test evaluate returns acceptable=True on exception."""
        evaluator = MagicMock()
        evaluator.generate_response.side_effect = Exception("Evaluation failed")

        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=evaluator,
        )

        evaluation = factory.evaluate("Hello", "Hi there!", [])

        assert evaluation.is_acceptable is True

    def test_chat_with_evaluator_passes_first(self, mock_chat_model, mock_evaluator_model):
        """Test chat with evaluator that passes on first try."""
        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=mock_evaluator_model,
        )

        result = factory.chat("Hello", [])

        assert result == "Test response"
        # Generator called once, evaluator called once
        assert mock_chat_model.generate_response.call_count == 1
        assert mock_evaluator_model.generate_response.call_count == 1

    def test_chat_with_evaluator_retries(self, mock_chat_model, mock_evaluator_model_rejects_then_accepts):
        """Test chat retries when evaluator rejects."""
        # Set up generator to return different responses
        mock_chat_model.generate_response.side_effect = ["First response", "Better response"]

        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=mock_evaluator_model_rejects_then_accepts,
            response_limit=5,
        )

        result = factory.chat("Hello", [])

        assert result == "Better response"
        # Generator called twice (initial + rerun)
        assert mock_chat_model.generate_response.call_count == 2

    def test_chat_with_evaluator_max_retries(self, mock_chat_model):
        """Test chat stops at response_limit."""
        # Evaluator always rejects
        evaluator = MagicMock()
        evaluator.generate_response.return_value = Evaluation(
            is_acceptable=False,
            feedback="Not good enough",
        )

        mock_chat_model.generate_response.return_value = "Response"

        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=evaluator,
            response_limit=3,
        )

        factory.chat("Hello", [])

        # Should try response_limit - 1 times after initial (initial + 2 reruns = 3 total)
        # evaluator called 2 times (response_limit - 1)
        assert evaluator.generate_response.call_count == 2

    def test_rerun_includes_feedback(self, mock_chat_model):
        """Test rerun includes feedback in system prompt."""
        factory = ChatFactory(generator_model=mock_chat_model)

        factory.rerun(
            reply="Original response",
            feedback="Make it longer",
            extended_history=[
                {"role": "system", "content": "Original prompt"},
                {"role": "user", "content": "Hello"},
            ],
        )

        # Check that generate_response was called with updated system prompt
        call_args = mock_chat_model.generate_response.call_args
        messages = call_args[1]["messages"]
        system_msg = messages[0]

        assert "Original response" in system_msg["content"]
        assert "Make it longer" in system_msg["content"]


# ============================================================================
# Streaming Tests
# ============================================================================


class TestChatFactoryStreaming:
    """Tests for ChatFactory streaming functionality."""

    def test_stream_chat_accumulate_true(self, mock_chat_model):
        """Test streaming with accumulate=True yields accumulated text."""
        factory = ChatFactory(generator_model=mock_chat_model)

        chunks = list(factory.stream_chat("Hello", [], accumulate=True))

        # With accumulate=True, each chunk should be accumulated
        assert chunks == ["chunk1", "chunk1chunk2", "chunk1chunk2chunk3"]

    def test_stream_chat_accumulate_false(self, mock_chat_model):
        """Test streaming with accumulate=False yields individual chunks."""
        factory = ChatFactory(generator_model=mock_chat_model)

        chunks = list(factory.stream_chat("Hello", [], accumulate=False))

        assert chunks == ["chunk1", "chunk2", "chunk3"]

    def test_stream_chat_exception(self, mock_chat_model):
        """Test streaming yields error message on exception."""
        mock_chat_model.stream_response.side_effect = Exception("Stream error")
        factory = ChatFactory(generator_model=mock_chat_model)

        chunks = list(factory.stream_chat("Hello", []))

        assert len(chunks) == 1
        assert "error" in chunks[0].lower()

    def test_get_stream_chat_returns_method(self, mock_chat_model):
        """Test get_stream_chat returns the stream_chat method."""
        factory = ChatFactory(generator_model=mock_chat_model)

        stream_fn = factory.get_stream_chat()

        assert stream_fn == factory.stream_chat
        assert callable(stream_fn)


# ============================================================================
# MCP Integration Tests
# ============================================================================


class TestChatFactoryMCPIntegration:
    """Tests for ChatFactory MCP integration."""

    def test_set_mcp_logging_level_valid(self, mock_chat_model, mock_mcp_client):
        """Test setting valid MCP logging level."""
        factory = ChatFactory(generator_model=mock_chat_model)
        factory.mcp_client = mock_mcp_client

        factory.set_mcp_logging_level("DEBUG")

        mock_mcp_client.set_logging_level.assert_called_once_with(level="debug")

    def test_set_mcp_logging_level_case_insensitive(self, mock_chat_model, mock_mcp_client):
        """Test logging level is case insensitive."""
        factory = ChatFactory(generator_model=mock_chat_model)
        factory.mcp_client = mock_mcp_client

        factory.set_mcp_logging_level("warning")

        mock_mcp_client.set_logging_level.assert_called_once_with(level="warning")

    def test_set_mcp_logging_level_invalid(self, mock_chat_model):
        """Test invalid logging level raises ValueError."""
        factory = ChatFactory(generator_model=mock_chat_model)

        with pytest.raises(ValueError) as exc_info:
            factory.set_mcp_logging_level("INVALID")

        assert "Invalid logging level" in str(exc_info.value)

    def test_set_mcp_logging_level_no_client(self, mock_chat_model):
        """Test setting logging level with no MCP client does not raise."""
        factory = ChatFactory(generator_model=mock_chat_model)

        # Should not raise
        factory.set_mcp_logging_level("INFO")

    def test_instantiate_prompt_no_client(self, mock_chat_model):
        """Test instantiate_prompt returns empty list with no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        result = factory.instantiate_prompt("test_prompt", lambda x: {})

        assert result == []

    def test_instantiate_resource_no_client(self, mock_chat_model):
        """Test instantiate_resource returns empty list with no MCP client."""
        factory = ChatFactory(generator_model=mock_chat_model)

        result = factory.instantiate_resource("test_resource", lambda x: {})

        assert result == []


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================


class TestChatFactoryEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_tools_list(self, mock_chat_model):
        """Test ChatFactory with empty tools list."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=[])

        assert factory.openai_tools == []
        assert factory.tool_map == {}

    def test_none_tools(self, mock_chat_model):
        """Test ChatFactory with tools=None."""
        factory = ChatFactory(generator_model=mock_chat_model, tools=None)

        assert factory.openai_tools == []
        assert factory.tool_map == {}

    def test_display_content_callback(self, mock_chat_model):
        """Test display_content callback is stored."""
        callback = MagicMock()
        factory = ChatFactory(generator_model=mock_chat_model, display_content=callback)

        assert factory.display_content == callback

    def test_tool_with_no_parameters(self, mock_chat_model):
        """Test tool with no parameters."""

        def no_params_tool() -> str:
            """A tool with no parameters.

            Args: None
            """
            return "result"

        factory = ChatFactory(generator_model=mock_chat_model, tools=[no_params_tool])

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "no_params_tool"
        tool_call.function.arguments = "{}"

        factory.handle_tool_call([tool_call])

        call_args = mock_chat_model.format_tool_result.call_args
        assert call_args[1]["result"] == "result"

    def test_response_limit_boundary(self, mock_chat_model):
        """Test response_limit of 1 means no retries."""
        evaluator = MagicMock()
        evaluator.generate_response.return_value = Evaluation(
            is_acceptable=False,
            feedback="Not good",
        )

        factory = ChatFactory(
            generator_model=mock_chat_model,
            evaluator_model=evaluator,
            response_limit=1,
        )

        factory.chat("Hello", [])

        # With response_limit=1, evaluator should not be called at all
        # because the while loop condition is responses < response_limit
        assert evaluator.generate_response.call_count == 0

    def test_kwargs_passed_to_generate_response(self, mock_chat_model):
        """Test generator_kwargs are passed to generate_response."""
        factory = ChatFactory(
            generator_model=mock_chat_model,
            generator_kwargs={"temperature": 0.5, "max_tokens": 100},
        )

        factory.chat("Hello", [])

        call_kwargs = mock_chat_model.generate_response.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
