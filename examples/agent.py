import os

import gradio as gr
from chat_factory import (
    ChatFactory,
    ChatModel,
)
from chat_factory.utils.factory_utils import configure_logging
from utils.gradio_mcp_helpers import (
    MCPHandler,
    create_mcp_input_components,
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
    configure_logging(level="WARNING")

    # Create ChatFactory instance
    chat_factory = ChatFactory(
        generator_model=openai_model,
        system_prompt=system_message,
        tools=tools,
        generator_kwargs={"reasoning_effort": "none"},
        mcp_config_path="utils/mcp_config.json",
    )
    chat_factory.set_mcp_logging_level(level="CRITICAL")
    chat = chat_factory.get_gradio_chat()

    with gr.Blocks() as demo:
        # Create explicit chatbot for MCP message injection
        chatbot = gr.Chatbot(label="Chat")
        gr.ChatInterface(fn=chat, chatbot=chatbot)

        # State to track current prompt/resource selection
        current_prompt_state: gr.State = gr.State(value={"name": None, "arguments": []})
        current_resource_state: gr.State = gr.State(value={"name": None, "uri": None, "variables": []})

        # MCP content display (for debugging/display purposes)
        mcp_content_display = gr.Textbox(
            label="MCP Content",
            lines=5,
            visible=False,
            interactive=False,
        )

        # Prompt input section
        with gr.Group(visible=False) as prompt_input_group:
            prompt_instructions, prompt_inputs, prompt_submit_btn = create_mcp_input_components()

        # Resource input section
        with gr.Group(visible=False) as resource_input_group:
            resource_instructions, resource_inputs, resource_submit_btn = create_mcp_input_components()

        # Main control row with dropdowns and exit button
        with gr.Row():
            prompts_dropdown = gr.Dropdown(
                choices=chat_factory.prompt_names if chat_factory.prompt_names else ["No prompts available"],
                label="Prompts",
                interactive=bool(chat_factory.prompt_names),
                scale=1,
            )
            resources_dropdown = gr.Dropdown(
                choices=chat_factory.resource_names if chat_factory.resource_names else ["No resources available"],
                label="Resources",
                interactive=bool(chat_factory.resource_names),
                scale=1,
            )
            exit_btn = gr.Button("Exit", variant="stop", scale=0)

        # Create MCP handler
        handler = MCPHandler(
            chat_factory=chat_factory,
        )

        # Build output lists for event handlers (chatbot is last element)
        prompt_outputs = [
            current_prompt_state,
            mcp_content_display,
            prompt_input_group,
            prompt_instructions,
            *prompt_inputs,
            prompt_submit_btn,
            chatbot,
        ]
        resource_outputs = [
            current_resource_state,
            mcp_content_display,
            resource_input_group,
            resource_instructions,
            *resource_inputs,
            resource_submit_btn,
            chatbot,
        ]

        # Connect event handlers (chatbot included in inputs and outputs)
        prompts_dropdown.change(
            fn=handler.on_prompt_selected,
            inputs=[prompts_dropdown, chatbot],
            outputs=prompt_outputs,
        )
        resources_dropdown.change(
            fn=handler.on_resource_selected,
            inputs=[resources_dropdown, chatbot],
            outputs=resource_outputs,
        )
        prompt_submit_btn.click(
            fn=handler.on_prompt_submit,
            inputs=[current_prompt_state, chatbot, *prompt_inputs],
            outputs=[mcp_content_display, prompt_input_group, chatbot],
        )
        resource_submit_btn.click(
            fn=handler.on_resource_submit,
            inputs=[current_resource_state, chatbot, *resource_inputs],
            outputs=[mcp_content_display, resource_input_group, chatbot],
        )
        exit_btn.click(fn=shutdown, outputs=gr.Textbox(visible=False))

    demo.launch()


if __name__ == "__main__":
    main()
