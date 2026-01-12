# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chat-factory** is a flexible Python framework for building LLM-powered chat applications with advanced capabilities:

- **Multi-provider LLM support**: OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama
- **Tool calling**: Custom Python functions as tools with automatic schema generation
- **MCP integration**: Connect to Model Context Protocol servers for external tools
- **Quality control**: Optional evaluator feedback loop for response validation
- **Structured output**: Type-safe responses using Pydantic models

**Use Cases**: AI chat assistants with custom capabilities, agents that call external APIs and tools, quality-controlled conversational AI, rapid prototyping with multiple LLM providers.

## Architecture Overview

The framework follows a layered design with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    ChatFactory                               │
│  (Orchestration: tool calling, evaluation, retry logic)     │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────────┐
│ChatModel│  │ Tools   │  │ SyncMCPClient│
│(LLM API)│  │(Custom) │  │(External)    │
└─────────┘  └─────────┘  └──────────────┘
    │             │             │
    ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────────┐
│Provider │  │ Schema  │  │ MCP Servers  │
│Clients  │  │ Utils   │  │(Background)  │
└─────────┘  └─────────┘  └──────────────┘
```

**Key Components:**

1. **ChatFactory** ([src/chat_factory/chat_factory.py](src/chat_factory/chat_factory.py)) - Orchestrates conversation flow with tool calling loops and optional evaluator-based quality control
2. **ChatModel** ([src/chat_factory/models.py](src/chat_factory/models.py)) - Unified interface for multiple LLM providers, handles provider-specific message formatting
3. **SyncMultiServerClient** ([src/chat_factory/sync_mcp_client.py](src/chat_factory/sync_mcp_client.py)) - Synchronous wrapper for async MCP client, runs in background thread with persistent event loop
4. **Schema Utilities** ([src/chat_factory/schema_utils.py](src/chat_factory/schema_utils.py)) - Automatic JSON schema generation from Python function signatures and docstrings

See [CHAT_FACTORY_ARCHITECTURE.md](CHAT_FACTORY_ARCHITECTURE.md) for comprehensive architecture details.

## Project Structure

```
/Users/antonio/AI/MyCode/chat-factory/
├── src/chat_factory/              # Core library code
│   ├── chat_factory.py           # ChatFactory orchestration
│   ├── async_chat_factory.py     # Async variant
│   ├── models.py                 # Multi-provider LLM interface
│   ├── sync_mcp_client.py        # Synchronous MCP client wrapper
│   ├── schema_utils.py           # Automatic schema generation
│   ├── mcp_utils.py              # MCP result processing
│   └── media_handler.py          # Media content handling
├── tests/                         # Comprehensive test suite
├── examples/                      # Usage examples (chat.py, agent.py, etc.)
├── docs/                          # Sphinx documentation
├── pyproject.toml                # Poetry configuration & dependencies
├── Makefile & run.sh             # Development commands
└── CHAT_FACTORY_ARCHITECTURE.md  # Detailed architecture docs
```

## Development Commands

This project uses Poetry for dependency management and a custom `run.sh` script for development tasks. All commands can be executed via either the Makefile (which delegates to `run.sh`) or directly via `run.sh`.

### Environment Setup
```bash
make venv                 # Create and activate local virtual environment
make install              # Install core dependencies
make install-lint         # Install linting dependencies
make install-test         # Install testing dependencies
make install-docs         # Install documentation dependencies
make install-dev          # Install all development dependencies (dev, test, lint, typing and docs dependency groups)
./run.sh install:all      # CI alternative: install all dependencies without interaction
```

### Code Quality
```bash
make format               # Format code with black and isort
make format-diff          # Run formatters on changed files
make lint                 # Run mypy, flake8, and pylint
make lint-diff            # Run all linters on changed files
make check                # Run format + lint + tests on all files (local development)
make pre-commit           # Format and lint only on changed files
./run.sh check:ci         # CI version (format only checks, no file modifications)
```

**Code Style Standards:**
- Line length: 119 characters
- Formatters: black, isort (profile: black)
- Linters: mypy (strict), flake8, pylint
- Type hints required for all functions

### Testing
```bash
make test                 # Run all tests
make test-cov             # Run tests with coverage
make coverage             # Generate coverage report
make test-verbose         # Run tests with verbose output
./run.sh tests:pattern "test_name"  # Run only tests matching pattern
```

**Test Organization:**
- [tests/](tests/) directory contains all tests
- [tests/conftest.py](tests/conftest.py) automatically adds src/ to Python path
- Tests cover: schema generation, tool registration, chat factory integration, error handling, OpenAI schema validation

### Documentation
```bash
make docs-api             # Generate API documentation automatically
make docs                 # Build Sphinx documentation
make docs-live            # Start live documentation server with auto-reload
make docs-clean           # Clean and rebuild documentation
```

### Package Building
```bash
make build                # Build package with Poetry
make validate-build       # Validate package builds correctly
make clean                # Clean build artifacts
```

## Key Development Patterns

### Adding a New LLM Provider

**For OpenAI-compatible APIs:**
1. Add entry to `OPENAI_CLIENT_MAP` in [src/chat_factory/models.py](src/chat_factory/models.py:31-42)
2. Include base_url and environment variable name

**For non-OpenAI-compatible APIs (like Anthropic):**
1. Add new client type in `ChatModel.__init__`
2. Add provider-specific logic in `generate_response()` method
3. Create conversion methods (e.g., `_prepare_messages_for_anthropic()`)
4. Handle provider differences (system parameters, tool formats, etc.)

### Working with Tools

Three flexible formats for registering custom tools:

```python
# 1. Auto-generation: Just pass functions
def my_tool(arg: str) -> dict:
    """Tool description.

    Args:
        arg: Argument description
    """
    return {"result": "value"}

factory = ChatFactory(
    generator_model=model,
    tools=[my_tool]  # Schema auto-generated
)

# 2. Hybrid: Function with description override
factory = ChatFactory(
    generator_model=model,
    tools=[
        {"function": my_tool, "description": "Custom description"}
    ]
)

# 3. Manual: Full control with explicit schema
factory = ChatFactory(
    generator_model=model,
    tools=[
        {
            "function": my_tool,
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": "Arg desc"}
                },
                "required": ["arg"]
            }
        }
    ]
)
```

**Schema Generation Guidelines:**
- Use Google-style docstrings with Args section
- Add type hints to all parameters
- [schema_utils.py](src/chat_factory/schema_utils.py) warns about missing type hints or descriptions

### MCP Integration

**Configuration** ([examples/mcp_config.json](examples/mcp_config.json)):
```json
{
  "mcpServers": {
    "server-name": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {"VARIABLE": "value"}
    }
  }
}
```

**Usage:**
```python
factory = ChatFactory(
    generator_model=model,
    tools=[...],  # Custom tools
    mcp_config_path="mcp_config.json"  # Adds MCP tools
)
```

The SyncMultiServerClient runs in a background thread and provides thread-safe synchronous access to async MCP operations.

### Testing Patterns

**Running specific tests:**
```bash
# Run all tests in a file
pytest tests/test_schema_generation.py

# Run tests matching a pattern
./run.sh tests:pattern "test_type_mapping"

# Run with verbose output
make test-verbose
```

**Adding new tests:**
1. Create test file in [tests/](tests/) directory
2. Use pytest fixtures from [tests/conftest.py](tests/conftest.py)
3. Follow existing naming convention: `test_*.py`
4. Test coverage for schema generation, tool registration, error handling

## Important Notes

### Sync vs Async Implementations

- **Synchronous**: [chat_factory.py](src/chat_factory/chat_factory.py) - Use with Gradio, synchronous apps
- **Asynchronous**: [async_chat_factory.py](src/chat_factory/async_chat_factory.py) - Use with FastAPI, async apps
- Logic is identical, execution model differs

### MCP Client Lifecycle Pattern

The SyncMultiServerClient uses a **long-running lifecycle task** to solve the "cancel scope in different task" error:
- Background thread maintains persistent event loop
- `_manage_client_lifecycle()` method keeps MCP client context alive
- Thread-safe operation via `asyncio.run_coroutine_threadsafe()`
- Always use context manager (`with` statement) or call `shutdown()` explicitly

### Provider-Specific Handling

**Anthropic differences:**
- System messages extracted to separate `system` parameter
- Structured output via tool calling (not native parse API)
- Tool calls in content blocks (not separate `tool_calls` field)
- Tool results use `user` role with `tool_result` content blocks

**OpenAI and compatible providers:**
- Standard message format with system messages in array
- Native structured output via `beta.chat.completions.parse()`
- Tool calls in separate field

### Evaluator Pattern

Optional quality control layer:
```python
factory = ChatFactory(
    generator_model=gpt4,
    evaluator_model=claude,  # Use different model to avoid bias
    response_limit=5  # Max retry attempts
)
```

The evaluator checks response quality and triggers regeneration with feedback if needed.

### Common Pitfalls

1. **Missing docstrings**: Auto-schema generation produces warnings and may have poor descriptions
2. **Tool name conflicts**: Custom tools override MCP tools with the same name
3. **Anthropic max_tokens**: Required parameter (defaults to 10000); OpenAI doesn't require it
4. **MCP client cleanup**: Always use context manager or call `shutdown()` to avoid resource leaks

## Additional Resources

- [CHAT_FACTORY_ARCHITECTURE.md](CHAT_FACTORY_ARCHITECTURE.md) - Comprehensive architecture documentation
- [examples/](examples/) - Practical usage examples (chat.py, agent.py, to_do.py)
- [docs/](docs/) - Sphinx documentation
- ReadTheDocs: https://chat-factory.readthedocs.io/
