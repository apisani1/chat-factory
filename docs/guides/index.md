# User Guides

This section contains comprehensive step-by-step guides for working with chat-factory.

## Installation

### Using pip

Install the latest version from PyPI:

```bash
pip install chat-factory
```

### Using Poetry

If you use Poetry for dependency management:

```bash
poetry add chat-factory
```

### Development Installation

To install for development with all dependencies:

```bash
git clone https://github.com/apisani1/chat-factory.git
cd chat-factory
make install-dev
```

### Requirements

- Python 3.10 or higher
- API keys for the LLM providers you want to use (OpenAI, Anthropic, etc.)

## Basic Usage

### Simple Chat

The most basic usage creates a chat interface with a single LLM:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Initialize the model
model = ChatModel("gpt-4o", provider="openai")

# Create a chat factory
factory = ChatFactory(generator_model=model)

# Chat with conversation history
history = []
response = factory.chat("Hello! Tell me a joke.", history)
print(response)

# Continue the conversation
history.append({"role": "user", "content": "Hello! Tell me a joke."})
history.append({"role": "assistant", "content": response})
response = factory.chat("Tell me another one!", history)
print(response)
```

### Multi-Turn Conversations

Maintain conversation history for context-aware responses:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

model = ChatModel("gpt-4o", provider="openai")
factory = ChatFactory(generator_model=model)

history = []

def chat_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = factory.chat(user_input, history)
        print(f"AI: {response}")

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

chat_loop()
```

### Using Different Providers

Switch between LLM providers easily:

```python
from chat_factory.models import ChatModel

# OpenAI
gpt4 = ChatModel("gpt-4o", provider="openai")
gpt_mini = ChatModel("gpt-4o-mini", provider="openai")

# Anthropic
claude = ChatModel("claude-sonnet-4", provider="anthropic")
claude_haiku = ChatModel("claude-3-5-haiku-20241022", provider="anthropic")

# Google
gemini = ChatModel("gemini-2.0-flash-exp", provider="google")

# DeepSeek
deepseek = ChatModel("deepseek-chat", provider="deepseek")

# Groq
groq = ChatModel("llama-3.3-70b-versatile", provider="groq")

# Ollama (local)
llama = ChatModel("llama3.3", provider="ollama")
```

## Tool Integration

### Custom Tools with Auto-Schema Generation

The easiest way to add tools is to use automatic schema generation:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

def get_weather(location: str) -> dict:
    """Get current weather for a location.

    Args:
        location: City name or coordinates
    """
    # Your weather API logic here
    return {
        "temperature": 72,
        "condition": "sunny",
        "humidity": 45,
        "location": location
    }

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculate tip amount for a bill.

    Args:
        bill_amount: Total bill amount in dollars
        tip_percentage: Tip percentage (default 15%)
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {
        "bill": bill_amount,
        "tip": tip,
        "total": total,
        "percentage": tip_percentage
    }

# Register tools - schemas auto-generated from signatures and docstrings
model = ChatModel("gpt-4o", provider="openai")
factory = ChatFactory(
    generator_model=model,
    tools=[get_weather, calculate_tip]
)

# The AI can now use these tools
history = []
response = factory.chat("What's the weather in New York?", history)
print(response)

response = factory.chat("Calculate a 20% tip on a $85 bill", history)
print(response)
```

### Tool Schema Formats

chat-factory supports three formats for tool registration:

**1. Auto-generation (Recommended)**

Just pass the function - schema is generated automatically:

```python
def my_tool(arg: str) -> dict:
    """Tool description.

    Args:
        arg: Argument description
    """
    return {"result": "value"}

factory = ChatFactory(generator_model=model, tools=[my_tool])
```

**2. Hybrid (Custom Description)**

Override the description while auto-generating parameters:

```python
factory = ChatFactory(
    generator_model=model,
    tools=[{
        "function": my_tool,
        "description": "Custom description that overrides docstring"
    }]
)
```

**3. Manual (Full Control)**

Provide the complete schema manually:

```python
factory = ChatFactory(
    generator_model=model,
    tools=[{
        "function": my_tool,
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                "arg": {
                    "type": "string",
                    "description": "Argument description"
                }
            },
            "required": ["arg"],
            "additionalProperties": False
        }
    }]
)
```

### Best Practices for Tools

1. **Use type hints**: Required for auto-schema generation
2. **Write clear docstrings**: Use Google-style format with Args section
3. **Return structured data**: Return dicts or Pydantic models
4. **Handle errors gracefully**: Return error messages in the dict
5. **Keep tools focused**: Each tool should do one thing well

Example of a well-designed tool:

```python
from typing import Optional

def search_database(
    query: str,
    limit: int = 10,
    category: Optional[str] = None
) -> dict:
    """Search the product database.

    This tool searches for products matching the query string.
    Results can be filtered by category and limited in number.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default 10)
        category: Optional category filter (e.g., "electronics", "books")

    Returns:
        Dictionary with search results
    """
    try:
        # Your search logic here
        results = perform_search(query, limit, category)
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

## MCP Integration

### What is MCP?

The Model Context Protocol (MCP) allows you to connect AI applications to external data sources and tools through a standardized interface. MCP servers can provide:

- File system access
- Web search capabilities
- Database queries
- API integrations
- Custom tools and resources

### Setting Up MCP

**1. Create an MCP configuration file** (`mcp_config.json`):

```json
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
        "BRAVE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**2. Use MCP in your chat factory**:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

model = ChatModel("claude-sonnet-4", provider="anthropic")
factory = ChatFactory(
    generator_model=model,
    mcp_config_path="mcp_config.json"
)

# Now the AI can use MCP tools
history = []
response = factory.chat("Search for information about Python asyncio", history)
print(response)
```

### Combining Custom Tools and MCP

You can use both custom tools and MCP tools together:

```python
def calculate_interest(principal: float, rate: float, years: int) -> dict:
    """Calculate compound interest.

    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as percentage)
        years: Number of years
    """
    amount = principal * ((1 + rate/100) ** years)
    return {
        "principal": principal,
        "rate": rate,
        "years": years,
        "final_amount": amount,
        "interest_earned": amount - principal
    }

factory = ChatFactory(
    generator_model=model,
    tools=[calculate_interest],  # Custom tool
    mcp_config_path="mcp_config.json"  # Plus MCP tools
)
```

### Available MCP Servers

Popular MCP servers you can use:

- `@modelcontextprotocol/server-filesystem` - File system access
- `@modelcontextprotocol/server-brave-search` - Web search
- `@modelcontextprotocol/server-github` - GitHub integration
- `@modelcontextprotocol/server-postgres` - PostgreSQL database
- `@modelcontextprotocol/server-slack` - Slack integration

See the [MCP server directory](https://github.com/modelcontextprotocol/servers) for more.

## Streaming Responses

For better user experience, stream responses as they're generated:

### Async Streaming

```python
import asyncio
from chat_factory import AsyncChatFactory
from chat_factory.models import ChatModel

async def main():
    model = ChatModel("gpt-4o", provider="openai")
    factory = AsyncChatFactory(generator_model=model)

    history = []
    async for chunk in factory.stream_chat("Tell me a story about a robot", history):
        print(chunk, end="", flush=True)
    print()  # New line at end

asyncio.run(main())
```

### Streaming with Gradio

```python
import gradio as gr
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

model = ChatModel("gpt-4o", provider="openai")
factory = ChatFactory(generator_model=model)

# Get Gradio-compatible streaming chat function
chat_fn = factory.get_stream_chat()

# Create Gradio interface
demo = gr.ChatInterface(
    chat_fn,
    title="Streaming Chat",
    examples=["Tell me a joke", "Explain quantum computing"]
)

demo.launch()
```

## Quality Control with Evaluators

Use a separate model to evaluate and improve response quality:

```python
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

# Use different models to avoid bias
generator = ChatModel("gpt-4o", provider="openai")
evaluator = ChatModel("claude-sonnet-4", provider="anthropic")

factory = ChatFactory(
    generator_model=generator,
    evaluator_model=evaluator,
    response_limit=5  # Max retry attempts
)

history = []
response = factory.chat("Explain machine learning in simple terms", history)
# Response will be regenerated if evaluator finds issues
print(response)
```

### How the Evaluator Works

1. Generator creates a response
2. Evaluator checks response quality
3. If rejected, generator tries again with feedback
4. Process repeats up to `response_limit` times
5. Best response is returned

This ensures higher quality responses but increases latency and token usage.

## Structured Output

Get type-safe responses using Pydantic models:

```python
from pydantic import BaseModel, Field
from chat_factory.models import ChatModel

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief review summary")
    pros: list[str] = Field(description="List of positives")
    cons: list[str] = Field(description="List of negatives")

model = ChatModel("gpt-4o", provider="openai")

messages = [
    {"role": "user", "content": "Review the movie 'The Matrix'"}
]

review = model.generate_response(
    messages,
    response_format=MovieReview
)

# review is a MovieReview instance
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}")
print(f"Pros: {', '.join(review.pros)}")
print(f"Cons: {', '.join(review.cons)}")
```

## Advanced Features

### Async Usage

For async applications (like FastAPI):

```python
import asyncio
from chat_factory import AsyncChatFactory
from chat_factory.models import ChatModel

async def main():
    model = ChatModel("gpt-4o", provider="openai")
    factory = AsyncChatFactory(generator_model=model)

    history = []
    response = await factory.chat("Hello!", history)
    print(response)

asyncio.run(main())
```

### Custom System Prompts

Set a system prompt to guide the AI's behavior:

```python
from chat_factory.models import ChatModel

model = ChatModel("gpt-4o", provider="openai")

messages = [
    {
        "role": "system",
        "content": "You are a helpful coding assistant specializing in Python."
    },
    {
        "role": "user",
        "content": "How do I read a CSV file?"
    }
]

response = model.generate_response(messages)
print(response)
```

### Provider-Specific Parameters

Pass provider-specific parameters:

```python
from chat_factory.models import ChatModel

# OpenAI with custom parameters
model = ChatModel("gpt-4o", provider="openai")
response = model.generate_response(
    messages,
    temperature=0.7,
    max_tokens=500,
    top_p=0.9
)

# Anthropic with custom parameters
model = ChatModel("claude-sonnet-4", provider="anthropic")
response = model.generate_response(
    messages,
    temperature=0.8,
    max_tokens=2000
)
```

### Gradio Integration

Create web UIs with Gradio:

```python
import gradio as gr
from chat_factory import ChatFactory
from chat_factory.models import ChatModel

model = ChatModel("gpt-4o", provider="openai")
factory = ChatFactory(generator_model=model)

# Get Gradio-compatible chat function
chat_fn = factory.get_chat()

# Create interface
demo = gr.ChatInterface(
    chat_fn,
    title="My AI Assistant",
    description="Chat with GPT-4o",
    examples=[
        "Tell me a joke",
        "Explain quantum computing",
        "Write a haiku about programming"
    ]
)

demo.launch(share=True)
```

## Examples

The [examples/](https://github.com/apisani1/chat-factory/tree/main/examples) directory contains many working examples:

### Command-Line Examples

- `stdio_chat.py` - Basic chat interface
- `stdio_agent.py` - Agent with custom tools
- `stdio_streaming_chat.py` - Streaming responses
- `stdio_async_chat.py` - Async chat
- `stdio_async_agent.py` - Async agent

### Gradio Web UI Examples

- `gradio_chat.py` - Basic web chat
- `gradio_agent.py` - Web agent with tools
- `gradio_streaming_chat.py` - Streaming web chat
- `gradio_async_chat.py` - Async web chat

### MCP Server Examples

- `examples/mcp_servers/tool_server.py` - Custom MCP tool server
- `examples/mcp_servers/resource_server.py` - MCP resource server
- `examples/mcp_servers/prompt_server.py` - MCP prompt server
- `examples/mcp_servers/todo_server.py` - Todo list MCP server

Run any example:

```bash
python examples/stdio_chat.py
```

## Troubleshooting

### API Key Issues

Make sure your API keys are set in environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

Or use a `.env` file:

```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
```

### Tool Schema Warnings

If you see warnings about missing type hints or docstrings, add them:

```python
# Bad - will produce warnings
def my_tool(arg):
    return {"result": arg}

# Good - properly typed and documented
def my_tool(arg: str) -> dict:
    """Process the argument.

    Args:
        arg: The input string to process
    """
    return {"result": arg}
```

### MCP Connection Issues

If MCP servers fail to connect:

1. Check that the MCP server command is correct
2. Verify environment variables are set
3. Ensure the MCP server is installed (`npx` command works)
4. Check file paths are absolute, not relative

### Common Pitfalls

1. **Forgetting to update history**: Always append messages to history for context
2. **Tool name conflicts**: MCP tools override custom tools with the same name
3. **Missing max_tokens for Anthropic**: Anthropic requires max_tokens parameter
4. **Not calling MCP client shutdown**: Use context manager or call `shutdown()`

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Architecture Guide](https://github.com/apisani1/chat-factory/blob/main/CHAT_FACTORY_ARCHITECTURE.md) - Deep dive into architecture
- [Examples](https://github.com/apisani1/chat-factory/tree/main/examples) - Working code examples
- [GitHub](https://github.com/apisani1/chat-factory) - Source code and issues
