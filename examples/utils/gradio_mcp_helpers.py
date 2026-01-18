"""Gradio helper functions for MCP prompts and resources UI."""

from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import gradio as gr
from chat_factory import (
    AsyncChatFactory,
    ChatFactory,
)
from chat_factory.utils.mcp_utils import (
    search_prompt,
    search_resource,
)


# Maximum number of arguments/variables supported in the UI
MAX_MCP_INPUTS = 5


def format_messages_for_display(messages: List[Dict[str, Any]]) -> str:
    """Format OpenAI-style messages for display.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.

    Returns:
        Formatted string representation.
    """
    if not messages:
        return "No content retrieved."

    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multimodal content
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    text_parts.append("[Image content]")
            content = "\n".join(text_parts)
        parts.append(f"**{role.capitalize()}:** {content}")

    return "\n\n".join(parts)


def create_mcp_input_components() -> Tuple[gr.Markdown, List[gr.Textbox], gr.Button]:
    """Create pre-allocated input components for MCP prompts/resources.

    Must be called within a gr.Blocks() context.

    Returns:
        Tuple of (instructions_markdown, input_textboxes_list, submit_button)
    """
    instructions = gr.Markdown(value="", visible=False)
    inputs = []
    for i in range(MAX_MCP_INPUTS):
        tb = gr.Textbox(
            label=f"Argument {i + 1}",
            visible=False,
            interactive=True,
        )
        inputs.append(tb)
    submit_btn = gr.Button("Submit", variant="primary", visible=False)
    return instructions, inputs, submit_btn


def build_prompt_input_updates(prompt_arguments: List[Any]) -> List[Any]:
    """Build gr.update() calls for prompt argument inputs.

    Args:
        prompt_arguments: List of PromptArgument objects with name, description, required.

    Returns:
        List of gr.update() calls: [instructions_update, input1..inputN_updates, submit_update]
    """
    if not prompt_arguments:
        # No arguments - hide everything
        updates: List[Any] = [gr.update(visible=False)]
        updates.extend([gr.update(visible=False, value="") for _ in range(MAX_MCP_INPUTS)])
        updates.append(gr.update(visible=False))
        return updates

    # Build instructions
    lines = ["**Please provide the following arguments:**\n"]
    for arg in prompt_arguments:
        req = " *(required)*" if arg.required else " *(optional)*"
        desc = f": {arg.description}" if arg.description else ""
        lines.append(f"- **{arg.name}**{req}{desc}")
    instructions_text = "\n".join(lines)

    updates = [gr.update(visible=True, value=instructions_text)]

    # Show inputs for each argument
    for i in range(MAX_MCP_INPUTS):
        if i < len(prompt_arguments):
            arg = prompt_arguments[i]
            label = f"{arg.name} {'(required)' if arg.required else '(optional)'}"
            updates.append(gr.update(visible=True, value="", label=label))
        else:
            updates.append(gr.update(visible=False, value=""))

    # Show submit button
    updates.append(gr.update(visible=True))
    return updates


def build_resource_input_updates(variables: List[str]) -> List[Any]:
    """Build gr.update() calls for resource variable inputs.

    Args:
        variables: List of variable names from URI template.

    Returns:
        List of gr.update() calls: [instructions_update, input1..inputN_updates, submit_update]
    """
    if not variables:
        # No variables - hide everything
        updates: List[Any] = [gr.update(visible=False)]
        updates.extend([gr.update(visible=False, value="") for _ in range(MAX_MCP_INPUTS)])
        updates.append(gr.update(visible=False))
        return updates

    # Build instructions
    lines = ["**Please provide the following variables:**\n"]
    for var in variables:
        lines.append(f"- **{var}**")
    instructions_text = "\n".join(lines)

    result: List[Any] = [gr.update(visible=True, value=instructions_text)]

    # Show inputs for each variable
    for i in range(MAX_MCP_INPUTS):
        if i < len(variables):
            result.append(gr.update(visible=True, value="", label=variables[i]))
        else:
            result.append(gr.update(visible=False, value=""))

    # Show submit button
    result.append(gr.update(visible=True))
    return result


def hide_all_inputs() -> List[Any]:
    """Return gr.update() calls to hide all input components.

    Returns:
        List of gr.update() calls to hide instructions, all inputs, and submit button.
    """
    updates: List[Any] = [gr.update(visible=False)]  # instructions
    updates.extend([gr.update(visible=False, value="") for _ in range(MAX_MCP_INPUTS)])
    updates.append(gr.update(visible=False))  # submit button
    return updates


def collect_arguments_from_inputs(prompt_arguments: List[Any], input_values: List[str]) -> Dict[str, str]:
    """Collect argument values from input fields into a dictionary.

    Args:
        prompt_arguments: List of PromptArgument objects.
        input_values: List of string values from input textboxes.

    Returns:
        Dictionary mapping argument names to values.
    """
    result: Dict[str, str] = {}
    for i, arg in enumerate(prompt_arguments):
        if i < len(input_values) and input_values[i]:
            result[arg.name] = input_values[i]
    return result


def collect_variables_from_inputs(variables: List[str], input_values: List[str]) -> Dict[str, str]:
    """Collect variable values from input fields into a dictionary.

    Args:
        variables: List of variable names.
        input_values: List of string values from input textboxes.

    Returns:
        Dictionary mapping variable names to values.
    """
    result: Dict[str, str] = {}
    for i, var in enumerate(variables):
        if i < len(input_values) and input_values[i]:
            result[var] = input_values[i]
    return result


# Response builder helpers for handler methods
def _empty_prompt_response() -> Tuple[Any, ...]:
    """Build response for invalid/unavailable prompt selection."""
    return (
        {"name": None, "arguments": []},
        gr.update(visible=False),
        gr.update(visible=False),
        *hide_all_inputs(),
    )


def _empty_resource_response() -> Tuple[Any, ...]:
    """Build response for invalid/unavailable resource selection."""
    return (
        {"name": None, "uri": None, "variables": []},
        gr.update(visible=False),
        gr.update(visible=False),
        *hide_all_inputs(),
    )


def _prompt_content_response(content: str) -> Tuple[Any, ...]:
    """Build response showing prompt content (no arguments needed)."""
    return (
        {"name": None, "arguments": []},
        gr.update(visible=True, value=content),
        gr.update(visible=False),
        *hide_all_inputs(),
    )


def _resource_content_response(content: str) -> Tuple[Any, ...]:
    """Build response showing resource content (no variables needed)."""
    return (
        {"name": None, "uri": None, "variables": []},
        gr.update(visible=True, value=content),
        gr.update(visible=False),
        *hide_all_inputs(),
    )


def _prompt_input_response(prompt_name: str, prompt_arguments: List[Any]) -> Tuple[Any, ...]:
    """Build response showing input fields for prompt arguments."""
    return (
        {"name": prompt_name, "arguments": list(prompt_arguments)},
        gr.update(visible=False),
        gr.update(visible=True),
        *build_prompt_input_updates(list(prompt_arguments)),
    )


def _resource_input_response(resource_name: str, uri: str, variables: List[str]) -> Tuple[Any, ...]:
    """Build response showing input fields for resource variables."""
    return (
        {"name": resource_name, "uri": uri, "variables": variables},
        gr.update(visible=False),
        gr.update(visible=True),
        *build_resource_input_updates(variables),
    )


def _submit_empty_response() -> Tuple[Any, ...]:
    """Build response for empty submit (no name in state)."""
    return (
        gr.update(visible=False),
        gr.update(visible=False),
    )


def _submit_content_response(content: str) -> Tuple[Any, ...]:
    """Build response showing submitted content."""
    return (
        gr.update(visible=True, value=content),
        gr.update(visible=False),
    )


class MCPHandler:
    """Handler for MCP prompt/resource interactions in sync Gradio apps."""

    def __init__(self, chat_factory: ChatFactory):
        """Initialize handler.

        Args:
            chat_factory: ChatFactory instance for instantiating prompts/resources.
        """
        self.chat_factory = chat_factory

    def on_prompt_selected(self, prompt_name: str) -> Tuple[Any, ...]:
        """Handle prompt selection - show content directly or show input fields."""
        if not prompt_name or prompt_name == "No prompts available":
            return _empty_prompt_response()

        prompt, prompt_arguments = search_prompt(self.chat_factory.mcp_prompts, prompt_name)
        if not prompt:
            return _empty_prompt_response()

        if not prompt_arguments:
            result = self.chat_factory.instantiate_prompt(prompt_name, lambda _: {})
            content = format_messages_for_display(result) if result else "No content"
            return _prompt_content_response(content)

        return _prompt_input_response(prompt_name, prompt_arguments)

    def on_prompt_submit(self, state: Dict[str, Any], *input_values: str) -> Tuple[Any, ...]:
        """Handle prompt submit - collect values and instantiate prompt."""
        if not state.get("name"):
            return _submit_empty_response()

        arguments = collect_arguments_from_inputs(state["arguments"], list(input_values))
        result = self.chat_factory.instantiate_prompt(state["name"], lambda _: arguments)
        content = format_messages_for_display(result) if result else "No content"
        return _submit_content_response(content)

    def on_resource_selected(self, resource_name: str) -> Tuple[Any, ...]:
        """Handle resource selection - show content directly or show input fields."""
        if not resource_name or resource_name == "No resources available":
            return _empty_resource_response()

        resource, uri, variables = search_resource(self.chat_factory.mcp_resources, resource_name)
        if not resource:
            return _empty_resource_response()

        if not variables:
            result = self.chat_factory.instantiate_resource(resource_name, lambda _: {})
            content = format_messages_for_display(result) if result else "No content"
            return _resource_content_response(content)

        return _resource_input_response(resource_name, uri, variables)  # type: ignore

    def on_resource_submit(self, state: Dict[str, Any], *input_values: str) -> Tuple[Any, ...]:
        """Handle resource submit - collect values and instantiate resource."""
        if not state.get("name"):
            return _submit_empty_response()

        variables = collect_variables_from_inputs(state["variables"], list(input_values))
        result = self.chat_factory.instantiate_resource(state["name"], lambda _: variables)
        content = format_messages_for_display(result) if result else "No content"
        return _submit_content_response(content)


class AsyncMCPHandler:
    """Handler for MCP prompt/resource interactions in async Gradio apps."""

    def __init__(self, chat_factory: AsyncChatFactory):
        """Initialize async handler.

        Args:
            chat_factory: AsyncChatFactory instance for instantiating prompts/resources.
        """
        self.chat_factory = chat_factory

    async def on_prompt_selected(self, prompt_name: str) -> Tuple[Any, ...]:
        """Handle prompt selection - show content directly or show input fields."""
        if not prompt_name or prompt_name == "No prompts available":
            return _empty_prompt_response()

        prompt, prompt_arguments = search_prompt(self.chat_factory.mcp_prompts, prompt_name)
        if not prompt:
            return _empty_prompt_response()

        if not prompt_arguments:
            result = await self.chat_factory.instantiate_prompt(prompt_name, lambda _: {})
            content = format_messages_for_display(result) if result else "No content"
            return _prompt_content_response(content)

        return _prompt_input_response(prompt_name, prompt_arguments)

    async def on_prompt_submit(self, state: Dict[str, Any], *input_values: str) -> Tuple[Any, ...]:
        """Handle prompt submit - collect values and instantiate prompt."""
        if not state.get("name"):
            return _submit_empty_response()

        arguments = collect_arguments_from_inputs(state["arguments"], list(input_values))
        result = await self.chat_factory.instantiate_prompt(state["name"], lambda _: arguments)
        content = format_messages_for_display(result) if result else "No content"
        return _submit_content_response(content)

    async def on_resource_selected(self, resource_name: str) -> Tuple[Any, ...]:
        """Handle resource selection - show content directly or show input fields."""
        if not resource_name or resource_name == "No resources available":
            return _empty_resource_response()

        resource, uri, variables = search_resource(self.chat_factory.mcp_resources, resource_name)
        if not resource:
            return _empty_resource_response()

        if not variables:
            result = await self.chat_factory.instantiate_resource(resource_name, lambda _: {})
            content = format_messages_for_display(result) if result else "No content"
            return _resource_content_response(content)

        return _resource_input_response(resource_name, uri, variables)  # type: ignore

    async def on_resource_submit(self, state: Dict[str, Any], *input_values: str) -> Tuple[Any, ...]:
        """Handle resource submit - collect values and instantiate resource."""
        if not state.get("name"):
            return _submit_empty_response()

        variables = collect_variables_from_inputs(state["variables"], list(input_values))
        result = await self.chat_factory.instantiate_resource(state["name"], lambda _: variables)
        content = format_messages_for_display(result) if result else "No content"
        return _submit_content_response(content)
