# API Reference

This section contains the complete API documentation for llm-chat-factory, automatically generated from the source code docstrings.

## Core Modules

### ChatFactory

The main orchestration class for creating chat applications:

- `ChatFactory` - Synchronous chat factory
- `AsyncChatFactory` - Asynchronous chat factory

See [chat_factory](chat_factory.rst) for complete details.

### ChatModel

Unified interface for multiple LLM providers:

- `ChatModel` - Main model class
- Provider support: OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama

See [chat_factory](chat_factory.rst) for complete details.

### Utilities

Helper modules for schema generation, MCP integration, and media handling:

- `schema_utils` - Automatic JSON schema generation from Python functions
- `mcp_utils` - MCP result processing and conversion
- `media_handler` - Media content handling

See [chat_factory.utils](chat_factory.utils.rst) for complete details.

## Complete API Documentation

For the complete, detailed API documentation including all classes, methods, and parameters, see:

```{toctree}
:maxdepth: 2

modules
```

## Quick Links

- [ChatFactory](chat_factory.rst#module-chat_factory.chat_factory)
- [AsyncChatFactory](chat_factory.rst#module-chat_factory.async_chat_factory)
- [ChatModel](chat_factory.rst#module-chat_factory.models)
- [Schema Utils](chat_factory.utils.rst#module-chat_factory.schema_utils)
- [MCP Utils](chat_factory.utils.rst#module-chat_factory.mcp_utils)
