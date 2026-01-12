import gradio as gr

from chat_factory import (
    ChatFactory,
    ChatModel,
)
from tools import tools


system_message = """
You are given a problem to solve, by using your todo tools to plan a list of steps, then carrying out each step in turn.
Now use the todo list tools, create a plan, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Markdown markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""


def main() -> None:
    openai_model = ChatModel(model_name="gpt-5.2", provider="openai")
    # anthropic_model = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
    # google_model = ChatModel(model_name="gemini-2.5-flash", provider="google")
    # deepseek_model = ChatModel(model_name="deepseek-chat", provider="deepseek")
    # groq_model = ChatModel(model_name="openai/gpt-oss-120b", provider="groq")
    # ollama_model = ChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")

    # Create ChatFactory instance
    chat = ChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
        tools=tools,
        generator_kwargs={"reasoning_effort": "none"},
        mcp_config_path="mcp_config.json",
    ).get_gradio_chat()

    # Pass the chat method to Gradio
    gr.ChatInterface(fn=chat).launch()


if __name__ == "__main__":
    main()
