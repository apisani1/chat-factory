import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import gradio as gr
from chat_factory import (
    AsyncChatFactory,
    AsyncChatModel,
)
from pypdf import PdfReader
from utils.tools import tools


reader = PdfReader("me/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Ed Donner"

ED_GENERATOR_PROMPT = f"""You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer, say so.
## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n
With this context, please chat with the user, always staying in character as {name}."""

ED_EVALUATOR_PROMPT = f"""You are an evaluator that decides whether a response to a question is acceptable.
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality.
The Agent is playing the role of {name} and is representing {name} on their website.
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website.
The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:"
## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n"
With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."""

chat_factory: Optional[AsyncChatFactory] = None
chat_fn = None
demo: Optional[gr.Blocks] = None

openai_model = AsyncChatModel(model_name="gpt-5.2", provider="openai")
anthropic_model = AsyncChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
# google_model = AsyncChatModel(model_name="gemini-2.5-flash", provider="google")
# deepseek_model = AsyncChatModel(model_name="deepseek-chat", provider="deepseek")
# groq_model = AsyncChatModel(model_name="openai/gpt-oss-120b", provider="groq")
# ollama_model = AsyncChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")


async def init_chat_factory() -> None:
    global chat_factory, chat_fn

    chat_factory = AsyncChatFactory(
        generator_model=openai_model,
        system_prompt=ED_GENERATOR_PROMPT,
        evaluator_model=anthropic_model,
        evaluator_system_prompt=ED_EVALUATOR_PROMPT,
        tools=tools,
        mcp_config_path="utils/mcp_config.json",
    )
    await chat_factory.connect_to_mcp_servers()
    chat_fn = chat_factory.get_async_gradio_chat()


async def shutdown() -> str:
    """Force exit the application."""

    # Do here any necessary cleanup before shutdown

    os._exit(0)
    return ""  # Never reached


async def chat(message: str, history: List[Dict[str, Any]]) -> str:
    # Wait until init_chat_factory has run
    # (Gradio guarantees demo.load runs before first call)
    return await chat_fn(message, history)  # type: ignore


def main() -> None:
    global demo

    # Do here any necessary setup before starting Gradio interface

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
