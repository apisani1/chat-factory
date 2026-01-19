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
from chat_factory.utils.factory import configure_logging
from pypdf import PdfReader
from examples.utils.gradio_mcp import (
    AsyncMCPHandler,
    convert_gradio_messages_to_openai,
    create_mcp_input_components,
)
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
handler: Optional[AsyncMCPHandler] = None
demo: Optional[gr.Blocks] = None
prompt_names: List[str] = []
resource_names: List[str] = []

# Shared state for OpenAI-format history (synchronized with MCP injections)
synced_state: Dict[str, List[Dict[str, Any]]] = {"openai_history": []}

openai_model = AsyncChatModel(model_name="gpt-5.2", provider="openai")
anthropic_model = AsyncChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
# google_model = AsyncChatModel(model_name="gemini-2.5-flash", provider="google")
# deepseek_model = AsyncChatModel(model_name="deepseek-chat", provider="deepseek")
# groq_model = AsyncChatModel(model_name="openai/gpt-oss-120b", provider="groq")
# ollama_model = AsyncChatModel(model_name="deepseek-r1:7b", provider="ollama", api_key="unused")


async def init_chat_factory() -> tuple:
    global chat_factory, chat_fn, handler, prompt_names, resource_names

    chat_factory = AsyncChatFactory(
        generator_model=openai_model,
        system_prompt=ED_GENERATOR_PROMPT,
        evaluator_model=anthropic_model,
        evaluator_system_prompt=ED_EVALUATOR_PROMPT,
        tools=tools,
        mcp_config_path="mcp_config.json",
    )
    await chat_factory.connect_to_mcp_servers()
    await chat_factory.set_mcp_logging_level(level="CRITICAL")
    chat_fn = chat_factory.get_async_gradio_chat()

    # Create handler after factory is ready
    handler = AsyncMCPHandler(
        chat_factory=chat_factory,
    )

    # Update dropdown choices after MCP connection
    prompt_names = chat_factory.prompt_names
    resource_names = chat_factory.resource_names

    prompts_choices = prompt_names if prompt_names else ["No prompts available"]
    resources_choices = resource_names if resource_names else ["No resources available"]

    return (
        gr.update(choices=prompts_choices, interactive=bool(prompt_names)),
        gr.update(choices=resources_choices, interactive=bool(resource_names)),
    )


async def shutdown() -> str:
    """Force exit the application."""

    # Do here any necessary cleanup before shutdown

    os._exit(0)
    return ""  # Never reached


async def chat(message: str, history: List[Dict[str, Any]]) -> str:
    """Chat using synchronized OpenAI-format history that includes MCP injections."""
    # Wait until init_chat_factory has run
    # (Gradio guarantees demo.load runs before first call)
    # Use the synced OpenAI history instead of ChatInterface's history
    openai_history = convert_gradio_messages_to_openai(synced_state["openai_history"])
    response = await chat_fn(message, openai_history)  # type: ignore
    # Update synced history with new exchange
    synced_state["openai_history"] = synced_state["openai_history"] + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return response


def main() -> None:
    global demo

    # Do here any necessary setup before starting Gradio interface
    configure_logging(level="WARNING")

    with gr.Blocks() as demo:
        # Create explicit chatbot for MCP message injection
        chatbot = gr.Chatbot(label="Chat")
        gr.ChatInterface(fn=chat, chatbot=chatbot)

        # State to track OpenAI-format history (for Gradio event system)
        history_state: gr.State = gr.State(value=[])

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
                choices=["Loading..."],
                label="Prompts",
                interactive=False,
                scale=1,
            )
            resources_dropdown = gr.Dropdown(
                choices=["Loading..."],
                label="Resources",
                interactive=False,
                scale=1,
            )
            exit_btn = gr.Button("Exit", variant="stop", scale=0)

        # Build output lists for event handlers (chatbot and history_state at end)
        prompt_outputs = [
            current_prompt_state,
            mcp_content_display,
            prompt_input_group,
            prompt_instructions,
            *prompt_inputs,
            prompt_submit_btn,
            chatbot,
            history_state,
        ]
        resource_outputs = [
            current_resource_state,
            mcp_content_display,
            resource_input_group,
            resource_instructions,
            *resource_inputs,
            resource_submit_btn,
            chatbot,
            history_state,
        ]

        # Wrapper functions to delegate to handler and sync state (initialized lazily)
        async def on_prompt_selected(
            prompt_name: str, current_history: List[Dict[str, Any]], current_openai_history: List[Dict[str, Any]]
        ) -> Any:
            if handler:
                result = await handler.on_prompt_selected(prompt_name, current_history, current_openai_history)
                synced_state["openai_history"] = result[-1]
                return result
            return None

        async def on_prompt_submit(
            state: Dict[str, Any],
            current_history: List[Dict[str, Any]],
            current_openai_history: List[Dict[str, Any]],
            *input_values: str,
        ) -> Any:
            if handler:
                result = await handler.on_prompt_submit(state, current_history, current_openai_history, *input_values)
                synced_state["openai_history"] = result[-1]
                return result
            return None

        async def on_resource_selected(
            resource_name: str, current_history: List[Dict[str, Any]], current_openai_history: List[Dict[str, Any]]
        ) -> Any:
            if handler:
                result = await handler.on_resource_selected(resource_name, current_history, current_openai_history)
                synced_state["openai_history"] = result[-1]
                return result
            return None

        async def on_resource_submit(
            state: Dict[str, Any],
            current_history: List[Dict[str, Any]],
            current_openai_history: List[Dict[str, Any]],
            *input_values: str,
        ) -> Any:
            if handler:
                result = await handler.on_resource_submit(
                    state, current_history, current_openai_history, *input_values
                )
                synced_state["openai_history"] = result[-1]
                return result
            return None

        # Connect event handlers (chatbot and history_state in inputs and outputs)
        prompts_dropdown.change(
            fn=on_prompt_selected,
            inputs=[prompts_dropdown, chatbot, history_state],
            outputs=prompt_outputs,
        )
        resources_dropdown.change(
            fn=on_resource_selected,
            inputs=[resources_dropdown, chatbot, history_state],
            outputs=resource_outputs,
        )
        prompt_submit_btn.click(
            fn=on_prompt_submit,
            inputs=[current_prompt_state, chatbot, history_state, *prompt_inputs],
            outputs=[mcp_content_display, prompt_input_group, chatbot, history_state],
        )
        resource_submit_btn.click(
            fn=on_resource_submit,
            inputs=[current_resource_state, chatbot, history_state, *resource_inputs],
            outputs=[mcp_content_display, resource_input_group, chatbot, history_state],
        )
        exit_btn.click(fn=shutdown, outputs=gr.Textbox(visible=False))

        # This runs once when Gradio server starts and
        # avoids creating an asyncio loop inside an existing one
        demo.load(init_chat_factory, outputs=[prompts_dropdown, resources_dropdown])

    demo.launch()


if __name__ == "__main__":
    main()
