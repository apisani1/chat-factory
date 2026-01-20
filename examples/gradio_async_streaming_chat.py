import os
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
)

import gradio as gr

from chat_factory import (
    AsyncChatFactory,
    AsyncChatModel,
)
from chat_factory.utils.factory import configure_logging
from utils.gradio_mcp import convert_gradio_messages_to_openai


system_message = """You are a helpful AI assistant.
Your responsibility is to provide accurate, professional, and engaging responses to user questions.
Be clear and concise in your answers."""

chat_factory: Optional[AsyncChatFactory] = None
chat_fn = None
demo: Optional[gr.Blocks] = None

openai_model = AsyncChatModel(model_name="gpt-4o-mini", provider="openai")
# anthropic_model = AsyncChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
# google_model = AsyncChatModel(model_name="gemini-2.5-flash", provider="google")
# deepseek_model = AsyncChatModel(model_name="deepseek-chat", provider="deepseek")
# groq_model = AsyncChatModel(model_name="openai/gpt-oss-120b", provider="groq")
# ollama_model = AsyncChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")


async def init_chat_factory() -> None:
    global chat_factory, chat_fn

    chat_factory = AsyncChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
    )
    await chat_factory.connect_to_mcp_servers()
    chat_fn = chat_factory.get_async_stream_chat()


async def shutdown() -> str:
    """Force exit the application."""

    # Do here any necessary cleanup before shutdown

    os._exit(0)
    return ""  # Never reached


async def chat(message: str, history: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """Wrapper that converts Gradio multimodal history to OpenAI format."""
    # Wait until init_chat_factory has run
    # (Gradio guarantees demo.load runs before first call)
    openai_history = convert_gradio_messages_to_openai(history)
    async for chunk in chat_fn(message, openai_history):  # type: ignore
        yield chunk


def main() -> None:
    global demo

    # Do here any necessary setup before starting Gradio interface
    configure_logging(level="INFO")

    with gr.Blocks() as demo:
        gr.ChatInterface(fn=chat)

        with gr.Row():
            exit_btn = gr.Button("Exit", variant="stop", scale=0)

        exit_btn.click(fn=shutdown, outputs=gr.Textbox(visible=False))

        # This runs once when Gradio server starts and
        # avoids creating an asyncio loop inside an existing one
        demo.load(init_chat_factory)

    demo.launch()


if __name__ == "__main__":
    main()
