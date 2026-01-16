import os

import gradio as gr

from chat_factory import (
    ChatFactory,
    ChatModel,
)
from utils.tools import tools


system_message = """You are a helpful AI assistant.
Your responsibility is to provide accurate, professional, and engaging responses to user questions.
Be clear and concise in your answers."""

chat_fn = None


def shutdown() -> str:
    """Force exit the application."""

    # Do here any necessary cleanup before shutdown

    os._exit(0)
    return ""  # Never reached


def main() -> None:
    openai_model = ChatModel(model_name="gpt-5-mini", provider="openai")
    # anthropic_model = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
    # google_model = ChatModel(model_name="gemini-2.5-flash", provider="google")
    # deepseek_model = ChatModel(model_name="deepseek-chat", provider="deepseek")
    # groq_model = ChatModel(model_name="openai/gpt-oss-120b", provider="groq")
    # ollama_model = ChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")

    # Do here any necessary setup before starting Gradio interface

    # Create ChatFactory and get streaming chat function
    chat = ChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
        tools=tools,
        mcp_config_path="utils/mcp_config.json",
    ).get_gradio_stream_chat()

    with gr.Blocks() as demo:
        gr.ChatInterface(fn=chat)

        with gr.Row():
            exit_btn = gr.Button("Exit", variant="stop", scale=0)

        exit_btn.click(fn=shutdown, outputs=gr.Textbox(visible=False))

    demo.launch()


if __name__ == "__main__":
    main()
