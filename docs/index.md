# llm-chat-factory Documentation

Welcome to the llm-chat-factory documentation! This is a flexible Python framework for building LLM-powered chat applications with advanced capabilities.

## What is llm-chat-factory?

**llm-chat-factory** is a comprehensive framework for building LLM-powered chatbot applications with:

- **Multi-provider LLM support**: OpenAI, Anthropic (Claude), Google Gemini, DeepSeek, Groq, Ollama
- **Tool calling**: Register custom Python functions as tools with automatic schema generation
- **MCP integration**: Connect to Model Context Protocol servers for external tools and data sources
- **Quality control**: Optional evaluator feedback loop for response validation and quality assurance
- **Structured output**: Type-safe responses using Pydantic models
- **Streaming support**: Real-time response streaming for better user experience
- **Sync & Async**: Both synchronous and asynchronous implementations for different use cases

## Why llm-chat-factory?

Building production-ready LLM applications involves many challenges:

- **Provider flexibility**: You want to switch between LLM providers without rewriting code
- **Tool integration**: You need to connect your AI to external functions, APIs, and data sources
- **Quality assurance**: You need to ensure responses meet your quality standards
- **Developer experience**: You want clean, simple APIs with sensible defaults

llm-chat-factory solves these challenges with a clean, extensible architecture that handles the complexity for you.

## Quick Example

Here's a simple example showing the power of llm-chat-factory:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Define a custom tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name or coordinates
    """
    return {"temp": 72, "condition": "sunny", "location": location}

# Initialize with automatic tool schema generation
model = ChatModel("gpt-5.2", provider="openai")
factory = ChatFactory(
    generator_model=model,
    tools=[get_weather]  # Schema auto-generated!
)

# The AI can now use your tool
history = []
response = factory.chat("What's the weather in San Francisco?", history)
print(response)
```

## Installation

Install via pip:

```bash
pip install llm-chat-factory
```

Or with Poetry:

```bash
poetry add llm-chat-factory
```

## Key Concepts

### ChatFactory

The `ChatFactory` and `AsyncChatFactory` classes orchestrate the entire conversation flow:

- Manage conversation history
- Executs tool calls (both custom and MCP tools)
- Optionally evaluates response quality
- Handles retry logic with feedback

### ChatModel

The `ChatModel` and `AsyncChatModel` classes provide a unified interface for all LLM providers:

- Single API across OpenAI, Anthropic, Google, etc.
- Automatic message format conversion
- Support for structured output via Pydantic
- Tool calling / function calling support

### Tools

Tools are Python functions that the AI can call:

- **Custom tools**: Your own Python functions
- **MCP tools**: External tools from MCP servers
- **Auto-schema generation**: Schemas generated from function signatures and docstrings

### MCP Integration

The Model Context Protocol (MCP) allows you to connect to external data sources and tools:

- File system access
- Web search
- Database queries
- Custom integrations

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ChatFactory                              │
│  (Orchestration: tool calling, evaluation, retry logic)     │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────────────────┐
│ChatModel│  │ Tools   │  │ SyncMultiServerClient│
│(LLM API)│  │(Custom) │  │     (External)       │
└─────────┘  └─────────┘  └──────────────────────┘
```

See the [Architecture Guide](https://github.com/apisani1/chat-factory/blob/main/CHAT_FACTORY_ARCHITECTURE.md) for comprehensive details.

## Use Cases

llm-chat-factory is perfect for:

- **AI Chat Assistants**: Build conversational AI with custom capabilities
- **AI Agents**: Create agents that interact with external tools and APIs
- **Quality-Controlled AI**: Applications requiring validated, high-quality responses
- **Multi-Provider Applications**: Applications that need flexibility to switch LLM providers
- **Rapid Prototyping**: Quickly test different models and approaches

## Next Steps

- [Installation Guide](guides/index.md#installation) - Get started with installation
- [Basic Usage](guides/index.md#basic-usage) - Learn the basics
- [Tool Integration](guides/index.md#tool-integration) - Add custom tools
- [MCP Integration](guides/index.md#mcp-integration) - Connect to external data
- [API Reference](api/index.md) - Complete API documentation

## Getting Help

- [GitHub Issues](https://github.com/apisani1/chat-factory/issues) - Report bugs or request features
- [GitHub Discussions](https://github.com/apisani1/chat-factory/discussions) - Ask questions
- [Examples](https://github.com/apisani1/chat-factory/tree/main/examples) - See working examples

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contents

Home <self>
Guides <guides/index>
API Reference <api/modules>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Useful Links

GitHub Repository <https://github.com/apisani1/chat-factory>
PyPI Package <https://pypi.org/project/llm-chat-factory/>
Issue Tracker <https://github.com/apisani1/chat-factory/issues>
