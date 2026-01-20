# Chat Factory Architecture Summary

This document describes the architecture of the `chat_factory` and `sync_mcp_client` modules, providing a comprehensive overview for onboarding in a new repository.

## Overview

The chat-factory project provides a flexible framework for building LLM-powered chat applications with:
- Multi-provider LLM support (OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama)
- Tool calling / function calling capabilities
- MCP (Model Context Protocol) integration for external tools
- Automatic schema generation from Python functions
- Evaluation loop for quality control

## Core Components

### 1. ChatFactory (`chat_factory.py`)

**Purpose**: Factory class for creating chat functions with optional evaluator feedback loop and MCP tool integration.

**Key Features**:
- Creates configurable chat interfaces with generator and optional evaluator models
- Supports custom Python function tools with automatic schema generation
- Integrates with MCP servers for external tool calling
- Implements retry loop with evaluator feedback for quality control
- Handles tool calling orchestration (both custom and MCP tools)

**Architecture**:
```
ChatFactory
├── Generator Model (ChatModel) - Generates responses
├── Evaluator Model (ChatModel, optional) - Evaluates response quality
├── Custom Tools (List[callable | dict]) - Python functions as tools
├── MCP Client (SyncMultiServerClient, optional) - External MCP tools
└── Tool Conversion System - Converts tools to OpenAI format
```

**Main Methods**:
- `__init__()`: Initializes factory with models, tools, and optional MCP config
- `chat(message, history)`: Main chat loop with tool calling and evaluation
- `_convert_tools_to_openai()`: Static method to convert custom tools to OpenAI format
- `get_chat()`: Returns chat function compatible with Gradio interface

**Tool Conversion System**:
Supports three formats for custom tools:
1. **Just function**: `[func1, func2]` - Auto-generates schema from signature and docstring
2. **Dict with auto-gen**: `[{"function": func1}]` - Auto-generates schema, optional description override
3. **Dict with manual**: `[{"function": func1, "parameters": {...}}]` - Backward compatible manual schema

**Chat Loop Flow**:
```
1. Receive user message + history
2. Generate response (with tool calling support)
3. While tool calls requested:
   a. Execute custom tools (from tool_map)
   b. Execute MCP tools (via mcp_client)
   c. Format results and append to conversation
   d. Generate next response
4. If evaluator enabled:
   a. Evaluate response quality
   b. If rejected: regenerate with feedback (up to response_limit attempts)
5. Return final response
```

**Dependencies**:
- `models.ChatModel`: Unified LLM interface
- `schema_utils.extract_function_schema`: Auto-generates JSON schemas from functions
- `mcp_utils.process_tool_result_content`: Converts MCP results to OpenAI format
- `sync_mcp_client.SyncMultiServerClient`: Synchronous MCP client wrapper
- `mcp_multi_server.utils.mcp_tools_to_openai_format`: Converts MCP tools to OpenAI format

---

### 2. SyncMultiServerClient (`sync_mcp_client.py`)

**Purpose**: Synchronous wrapper for the async MCP MultiServerClient, enabling easy integration with synchronous code.

**Key Features**:
- Runs MCP client in background thread with persistent event loop
- Provides thread-safe synchronous API for async operations
- Proper lifecycle management with context manager support
- Automatic cleanup on program exit via atexit hook
- Solves "cancel scope in different task" error via long-running lifecycle task

**Architecture**:
```
SyncMultiServerClient
├── Background Thread (daemon)
│   └── Event Loop (asyncio)
│       └── Lifecycle Task (_manage_client_lifecycle)
│           ├── __aenter__ (creates cancel scope in THIS task)
│           ├── Stay alive until shutdown
│           └── __aexit__ (exits in SAME task - no errors!)
│
├── MCP Client (MultiServerClient)
│   └── Connected to multiple MCP servers via config
│
└── Thread-Safe Methods
    ├── list_tools() - List available MCP tools
    ├── call_tool() - Execute tool (blocks until complete)
    └── shutdown() - Cleanup and stop
```

**Lifecycle Management**:

The key innovation is the `_manage_client_lifecycle()` long-running task that solves the async context manager problem:

```python
async def _manage_client_lifecycle(self):
    """Long-running task managing MCP client context lifecycle."""
    try:
        # Enter context (creates cancel scope in THIS task)
        self.mcp_client = MultiServerClient.from_config(self.config_path)
        await self.mcp_client.__aenter__()

        # Signal initialization complete
        self._init_complete.set()

        # Stay alive until shutdown requested
        # This keeps cancel scope alive in this task
        while not self._shutdown:
            await asyncio.sleep(0.1)

    finally:
        # Exit context in SAME task (no cancel scope error!)
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(None, None, None)
```

**Why This Design?**

Previous attempts failed because async context managers (`__aenter__`/`__aexit__`) create a cancel scope that must be entered and exited in the **same async task**. By using a long-running lifecycle task:
1. `__aenter__` creates the cancel scope in the lifecycle task
2. Task stays alive throughout client lifetime
3. `__aexit__` runs in the same task (no error!)

**Usage Patterns**:

```python
# Context manager (recommended)
with SyncMultiServerClient(config_path) as client:
    tools = client.list_tools()
    result = client.call_tool("tool_name", {"arg": "value"})

# Manual lifecycle management
client = SyncMultiServerClient(config_path)
tools = client.list_tools()
# ... use client ...
client.shutdown()  # Or let atexit handle it
```

**Thread Safety**:
All public methods use `asyncio.run_coroutine_threadsafe()` to schedule operations on the background event loop, ensuring thread-safe operation.

**Key Methods**:
- `__init__(config_path)`: Starts background thread, initializes MCP client, registers atexit cleanup
- `list_tools()`: Returns `ListToolsResult` with available MCP tools (synchronous access)
- `call_tool(name, args)`: Executes tool and returns `CallToolResult` (blocks until complete)
- `shutdown()`: Graceful cleanup (safe to call multiple times)

---

### 3. ChatModel (`models.py`)

**Purpose**: Unified interface for multiple LLM providers with support for text generation, structured output, and tool calling.

**Supported Providers**:
- OpenAI (including OpenAI-compatible APIs)
- Anthropic (Claude)
- Google Gemini
- DeepSeek
- Groq
- Ollama (local models)

**Key Features**:
- Single API for all providers
- Automatic provider-specific message formatting
- Structured output via Pydantic models
- Tool calling / function calling support
- Provider-specific optimizations (e.g., Anthropic's system parameter)

**Architecture**:
```
ChatModel
├── Provider Client (OpenAI | Anthropic)
├── Model Name (e.g., "gpt-4o", "claude-sonnet-4")
└── API Key / Configuration
```

**Provider-Specific Handling**:

**OpenAI**:
- Native structured output via `beta.chat.completions.parse()`
- Direct tool calling support
- Standard message format

**Anthropic**:
- System messages moved to separate `system` parameter
- Structured output via tool calling (not native parse API)
- Tool calls converted from `tool_calls` field to content blocks
- Tool results use `user` role with `tool_result` content blocks

**Main Methods**:
- `generate_response(messages, *, max_tokens, response_format, tools, **kwargs)`:
  - Returns `str` for text responses
  - Returns Pydantic model instance for structured responses
  - Returns `List[ChatCompletionMessageToolCall]` when tools are called
- `format_tool_result(tool_call_id, result)`: Formats tool results for conversation

**Response Modes**:
1. **Text mode**: No `response_format` or `tools` → returns string
2. **Structured response mode**: `response_format=PydanticModel` → returns model instance
3. **Tool calling mode**: `tools=[...]` → returns tool calls list (caller handles loop)

**Static Helper Methods**:
- `_prepare_messages_for_anthropic()`: Converts OpenAI format to Anthropic format
- `_prepare_tool_params()`: Prepares structured response tool for Anthropic
- `_convert_tools_to_anthropic()`: Converts OpenAI tool format to Anthropic
- `_convert_tool_calls_to_openai()`: Converts Anthropic tool_use blocks to OpenAI format

---

### 4. Schema Utilities (`schema_utils.py`)

**Purpose**: Automatic JSON schema generation from Python function signatures and docstrings.

**Key Features**:
- Extracts parameter types from type hints
- Parses Google-style docstrings for descriptions
- Generates OpenAI-compatible tool schemas
- Handles complex types (List[str], nested types, etc.)
- Provides helpful warnings for missing type hints or descriptions

**Main Function**:
```python
def extract_function_schema(func) -> dict:
    """Generate complete JSON schema from Python function.

    Returns:
        {
            "name": "function_name",
            "description": "Function description from docstring",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string",
                        "description": "Param description from docstring"
                    }
                },
                "required": ["param_name"],
                "additionalProperties": False
            }
        }
    """
```

**Type Mapping**:
- `str` → `"string"`
- `int` → `"integer"`
- `float` → `"number"`
- `bool` → `"boolean"`
- `list` / `List` → `"array"`
- `dict` / `Dict` → `"object"`
- `List[str]` → `{"type": "array", "items": {"type": "string"}}`

**Docstring Parsing**:
Expects Google-style docstrings:
```python
def example(name: str, count: int = 5):
    """Short description.

    Longer description if needed.

    Args:
        name: Description of name parameter
        count: Description of count parameter

    Returns:
        Description of return value
    """
```

**Warnings**:
- Missing type hints (defaults to "string")
- Missing docstring
- Missing parameter descriptions

---

### 5. MCP Utilities (`mcp_utils.py`)

**Purpose**: Utilities for working with MCP tool results and converting them to OpenAI-compatible formats.

**Key Functions**:

**`process_tool_result_content(tool_result: CallToolResult, verbose: bool = False) -> str`**:
- Converts MCP `CallToolResult` to string format suitable for OpenAI tool responses
- Handles multiple content types (text, image, audio, embedded resources, resource links)
- Images and audio converted to text descriptions (OpenAI tool messages must be text-only)
- Optionally displays content to user (with `verbose=True`)

**Content Type Handling**:
- `TextContent`: Extracts text directly
- `ImageContent`: Converts to `"[Image: {mimeType} received]"`
- `AudioContent`: Converts to `"[Audio: {mimeType} received]"`
- `EmbeddedResource`: Extracts text or creates description for binary
- `ResourceLink`: Creates link description

**Other Functions** (for MCP prompt/resource handling):
- `handle_content_block()`: Displays content to user (images shown, audio played)
- `search_and_instantiate_prompt()`: Retrieves and converts MCP prompts to OpenAI messages
- `search_and_instantiate_resource()`: Retrieves and converts MCP resources to messages

---

## Integration Flow

Here's how all components work together:

```
User Request
    ↓
ChatFactory.chat(message, history)
    ↓
ChatModel.generate_response(messages, tools=openai_tools)
    ↓
[Tool calls returned]
    ↓
ChatFactory.handle_tool_call(tool_calls)
    ├── Custom Tool? → Execute from tool_map
    └── MCP Tool? → SyncMultiServerClient.call_tool()
                        ↓
                    MCP Server (via background thread)
                        ↓
                    CallToolResult
                        ↓
                    mcp_utils.process_tool_result_content()
    ↓
Format results → Append to messages
    ↓
ChatModel.generate_response(messages, tools=openai_tools)
    ↓
[Final text response]
    ↓
[Optional: Evaluator checks quality, may trigger retry]
    ↓
Return to user
```

## Configuration

### MCP Configuration (`mcp_config.json`)

Example structure:
```json
{
  "mcpServers": {
    "server-name": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "VARIABLE": "value"
      }
    }
  }
}
```

### Tool Registration

Three ways to register custom tools with ChatFactory:

```python
# 1. Just functions (auto-schema generation)
def my_tool(arg: str) -> dict:
    """Tool description.

    Args:
        arg: Argument description
    """
    return {"result": "value"}

factory = ChatFactory(
    generator_model=model,
    tools=[my_tool]  # ← Auto-generates schema
)

# 2. Functions with description override
factory = ChatFactory(
    generator_model=model,
    tools=[
        {"function": my_tool, "description": "Custom description"}
    ]
)

# 3. Manual schema (backward compatible)
factory = ChatFactory(
    generator_model=model,
    tools=[
        {
            "function": my_tool,
            "description": "Tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": "Arg description"}
                },
                "required": ["arg"]
            }
        }
    ]
)
```

## Design Decisions

### 1. Why ChatFactory?
- Encapsulates complex chat logic (tool calling, evaluation, retry)
- Provides reusable abstraction for chat applications
- Separates concerns: ChatFactory (orchestration) vs ChatModel (LLM interface)

### 2. Why Separate ChatModel?
- Single interface for multiple providers reduces complexity
- Provider-specific logic isolated in one place
- Easy to add new providers
- Handles provider quirks (Anthropic system parameter, tool format differences)

### 3. Why SyncMultiServerClient?
- MCP client is async, but many applications are synchronous
- Background thread + event loop provides clean synchronous API
- Long-running lifecycle task solves cancel scope problem elegantly
- Thread-safe design allows use from any thread

### 4. Why Automatic Schema Generation?
- Reduces boilerplate and duplication
- Enforces documentation (warns on missing docstrings)
- Single source of truth (function signature)
- Still supports manual schemas for edge cases

### 5. Evaluator Pattern
- Optional quality control layer
- Can enforce business rules, safety checks, formatting requirements
- Configurable retry limit prevents infinite loops
- Uses separate model to avoid bias

## Testing Strategy

The codebase includes comprehensive tests:
- **Schema generation tests**: Type mapping, docstring parsing, edge cases
- **Tool registration tests**: All three tool formats, validation
- **Chat factory tests**: Tool calling, MCP integration, evaluator loop
- **Error handling tests**: Missing tools, malformed arguments, network errors
- **OpenAI schema validation tests**: Ensures generated schemas are valid

## Dependencies

**Core**:
- `anthropic`: Anthropic API client
- `openai`: OpenAI API client (also used for compatible APIs)
- `pydantic`: Data validation and structured output
- `python-dotenv`: Environment variable management

**MCP**:
- `mcp`: Model Context Protocol SDK
- `mcp-multi-server`: Multi-server MCP client implementation

**Optional**:
- `gradio`: For building chat UIs (if using `get_chat()`)

## Future Enhancements

Potential improvements:
1. **Streaming support**: Add streaming response generation
2. **Conversation persistence**: Save/load conversation history
3. **Tool result caching**: Cache deterministic tool results
4. **Parallel tool execution**: Execute independent tool calls in parallel
5. **Retry with exponential backoff**: Handle rate limits gracefully
6. **Token usage tracking**: Monitor and log token consumption
7. **Custom evaluator prompts**: Per-tool or per-conversation evaluator customization

## Common Pitfalls

1. **MCP client not properly shutdown**: Always use context manager or call `shutdown()`
2. **Tool name conflicts**: Custom tools and MCP tools with same name → MCP tool wins
3. **Missing docstrings**: Auto-schema generation warns but may produce poor descriptions
4. **Tool result format**: Must return dict or string; complex objects may cause issues
5. **Anthropic max_tokens**: Required parameter (defaults to 10000); OpenAI doesn't need it

## Example Usage

```python
from chat_factory import ChatFactory
from models import ChatModel

# Initialize models
generator = ChatModel("gpt-4o", provider="openai")
evaluator = ChatModel("gpt-4o-mini", provider="openai")

# Define custom tool
def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name or coordinates
    """
    return {"temp": 72, "condition": "sunny"}

# Create factory with MCP integration
factory = ChatFactory(
    generator_model=generator,
    evaluator_model=evaluator,
    tools=[get_weather],
    mcp_config_path="mcp_config.json"
)

# Chat
history = []
response = factory.chat("What's the weather in San Francisco?", history)
print(response)

# Continue conversation
history.append({"role": "user", "content": "What's the weather in San Francisco?"})
history.append({"role": "assistant", "content": response})
response = factory.chat("How about New York?", history)
```

---

## Summary

The chat-factory architecture provides:
- **Flexibility**: Multiple LLM providers, custom and MCP tools
- **Simplicity**: Clean APIs with sensible defaults
- **Robustness**: Comprehensive error handling and testing
- **Extensibility**: Easy to add providers, tools, or features
- **Quality**: Optional evaluator loop for response quality control

The design emphasizes separation of concerns, with each component handling a specific responsibility while maintaining clean interfaces between layers.
