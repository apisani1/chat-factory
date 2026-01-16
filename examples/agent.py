import os

import gradio as gr

from chat_factory import (
    ChatFactory,
    ChatModel,
)
from utils.tools import tools


system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Markdown markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""


def shutdown() -> str:
    """Force exit the application."""

    # Do here any necessary cleanup before shutdown

    os._exit(0)
    return ""  # Never reached


def main() -> None:
    openai_model = ChatModel(model_name="gpt-5.2", provider="openai")
    # anthropic_model = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
    # google_model = ChatModel(model_name="gemini-2.5-flash", provider="google")
    # deepseek_model = ChatModel(model_name="deepseek-chat", provider="deepseek")
    # groq_model = ChatModel(model_name="openai/gpt-oss-120b", provider="groq")
    # ollama_model = ChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")

    # Do here any necessary setup before starting Gradio interface

    # Create ChatFactory instance
    chat = ChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
        tools=tools,
        generator_kwargs={"reasoning_effort": "none"},
        mcp_config_path="utils/mcp_config.json",
    ).get_gradio_chat()

    with gr.Blocks() as demo:
        gr.ChatInterface(fn=chat)

        with gr.Row():
            exit_btn = gr.Button("Exit", variant="stop", scale=0)

        exit_btn.click(fn=shutdown, outputs=gr.Textbox(visible=False))

    demo.launch()


if __name__ == "__main__":
    main()
