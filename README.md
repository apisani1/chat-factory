# chat-factory

A flexible Python framework for building LLM-powered chat applications with advanced capabilities including multi-provider support, tool calling, MCP integration, and quality control.

[![Documentation Status](https://readthedocs.org/projects/chat-factory/badge/?version=latest)](https://chat-factory.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/chat-factory.svg)](https://pypi.org/project/chat-factory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**chat-factory** is a comprehensive framework for building LLM-powered chatbot applications with:

- **Multi-provider LLM support**: OpenAI, Anthropic (Claude), Google Gemini, DeepSeek, Groq, Ollama
- **Tool calling**: Custom Python functions as tools with automatic schema generation
- **MCP integration**: Connect to Model Context Protocol servers for external tools and data sources
- **Quality control**: Optional evaluator feedback loop for response validation
- **Structured output**: Type-safe responses using Pydantic models
- **Streaming support**: Real-time response streaming for better UX
- **Sync & Async**: Both synchronous and asynchronous implementations

## Use Cases

- AI chat assistants with custom capabilities
- Agents that call external APIs and tools
- Quality-controlled conversational AI
- Rapid prototyping with multiple LLM providers
- Integration with external data sources via MCP

## Installation

```bash
pip install chat-factory
```

Or, if you use Poetry:

```bash
poetry add chat-factory
```

## Quick Start

### Basic Chat

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Initialize the model
model = ChatModel("gpt-5.2", provider="openai")

# Create a chat factory
factory = ChatFactory(generator_model=model)

# Start chatting
history = []
response = factory.chat("Hello! What can you help me with?", history)
print(response)
```

### Chat with Custom Tools

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Define a custom tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name or coordinates
    """
    # Your weather API logic here
    return {"temp": 72, "condition": "sunny", "location": location}

# Initialize model and factory with tools
model = ChatModel("gpt-5.2", provider="openai")
factory = ChatFactory(
    generator_model=model,
    tools=[get_weather]  # Schema auto-generated from function signature
)

# The AI can now call your weather tool
history = []
response = factory.chat("What's the weather in San Francisco?", history)
print(response)
```

### Chat with MCP Integration

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Initialize with MCP configuration
model = ChatModel("claude-sonnet-4-5", provider="anthropic")
factory = ChatFactory(
    generator_model=model,
    mcp_config_path="mcp_config.json"  # Connects to MCP servers
)

# Now the AI can use MCP tools
history = []
response = factory.chat("Search for information about Python asyncio", history)
print(response)
```

### Chat with Quality Control

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Use different models for generation and evaluation
generator = ChatModel(model_name="gpt-5.2", provider="openai")
evaluator = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")

factory = ChatFactory(
    generator_model=generator,
    evaluator_model=evaluator,  # Evaluates response quality
    response_limit=5  # Max retry attempts
)

history = []
response = factory.chat("Explain quantum computing", history)
# Response will be regenerated if evaluator finds issues
print(response)
```

## Key Features

### 🤖 Multi-Provider Support

Switch between LLM providers with a single parameter:

```python
from chat_factory.models import ChatModel

# OpenAI
gpt4 = ChatModel("gpt-5.2", provider="openai")

# Anthropic
claude = ChatModel("claude-sonnet-4-5", provider="anthropic")

# Google
gemini = ChatModel("gemini-2.5-flash", provider="google")

# DeepSeek
deepseek = ChatModel("deepseek-chat", provider="deepseek")

# Groq
groq = ChatModel(model_name="openai/gpt-oss-120b", provider="groq")

# Local with Ollama
llama = ChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")
```

### 🔧 Automatic Tool Schema Generation

Three flexible formats for registering tools:

```python
# 1. Auto-generation: Just pass functions
def my_tool(arg: str) -> dict:
    """Tool description.

    Args:
        arg: Argument description
    """
    return {"result": "value"}

factory = ChatFactory(generator_model=model, tools=[my_tool])

# 2. Hybrid: Override description
factory = ChatFactory(
    generator_model=model,
    tools=[{"function": my_tool, "description": "Custom description"}]
)

# 3. Manual: Full control
factory = ChatFactory(
    generator_model=model,
    tools=[{
        "function": my_tool,
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "Arg desc"}
            },
            "required": ["arg"]
        }
    }]
)
```

### 🔌 MCP (Model Context Protocol) Integration

Connect to external tools and data sources:

```json
// mcp_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

```python
factory = ChatFactory(
    generator_model=model,
    mcp_config_path="mcp_config.json"
)
```

### 📊 Structured Output

Get type-safe responses using Pydantic models:

```python
from pydantic import BaseModel
from chat_factory.models import ChatModel

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: int

model = ChatModel("gpt-5.2", provider="openai")
response = model.generate_response(
    [{"role": "user", "content": "What's the weather in SF?"}],
    response_format=WeatherResponse
)
# response is a WeatherResponse instance
print(f"Temp: {response.temperature}°F")
```

### 🌊 Streaming Support

Stream responses in real-time:

```python
from chat_factory import AsyncChatFactory
from chat_factory.async_models import AsyncChatModel

model = AsyncChatModel("gpt-5.2", provider="openai")
factory = AsyncChatFactory(generator_model=model)

async for chunk in factory.astream_chat("Tell me a story", [], accumulate=False):
    print(chunk, end="", flush=True)
```

## Documentation

Full documentation is available at [https://chat-factory.readthedocs.io/](https://chat-factory.readthedocs.io/)

- [Installation Guide](https://chat-factory.readthedocs.io/en/latest/guides/index.html#installation)
- [Basic Usage](https://chat-factory.readthedocs.io/en/latest/guides/index.html#basic-usage)
- [Tool Integration](https://chat-factory.readthedocs.io/en/latest/guides/index.html#tool-integration)
- [MCP Integration](https://chat-factory.readthedocs.io/en/latest/guides/index.html#mcp-integration)
- [API Reference](https://chat-factory.readthedocs.io/en/latest/api/index.html)

## Examples

The [examples/](examples/) directory contains comprehensive examples:

- `stdio_chat.py` - Basic command-line chat
- `stdio_agent.py` - Agent with custom tools
- `stdio_streaming_chat.py` - Streaming chat responses
- `gradio_chat.py` - Web UI with Gradio
- `gradio_agent.py` - Web-based agent with tools
- MCP server examples in `examples/mcp_servers/`

Run an example:

```bash
python examples/stdio_chat.py
```

## Development

This project uses Poetry for dependency management and a set of development commands:

```bash
# Install dependencies
make install-dev

# Format code
make format

# Run linters
make lint

# Run tests
make test

# Build documentation
make docs-live
```

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## Architecture

The framework follows a layered design:

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
│(LLM API)│  │(Custom) │  │(External)            │
└─────────┘  └─────────┘  └──────────────────────┘
```

See [CHAT_FACTORY_ARCHITECTURE.md](CHAT_FACTORY_ARCHITECTURE.md) for comprehensive architecture details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- [Documentation](https://chat-factory.readthedocs.io/)
- [PyPI Package](https://pypi.org/project/chat-factory/)
- [GitHub Repository](https://github.com/apisani1/chat-factory)
- [Issue Tracker](https://github.com/apisani1/chat-factory/issues)
- [Release Notes](https://github.com/apisani1/chat-factory/releases)
