"""Gradio helper functions for MCP prompts and resources UI."""

import base64
import logging
import mimetypes
import os
import tempfile
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
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
from utils.media_handler import (
    get_audio,
    get_image,
)


logger = logging.getLogger(__name__)


def _save_base64_to_temp_file(b64_data: str, extension: str) -> str:
    """Decode base64 data and save to a temporary file.

    Args:
        b64_data: Base64-encoded data.
        extension: File extension (e.g., "png", "wav").

    Returns:
        Path to the temporary file.

    Raises:
        ValueError: If base64 decoding fails.
    """
    decoded = base64.b64decode(b64_data)
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
        tmp.write(decoded)
        return tmp.name


def convert_gradio_content_to_openai(content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
    """Convert Gradio-format content to OpenAI-format content.

    Gradio uses: {"type": "file", "file": {"path": "/path/to/image.png"}}
    OpenAI uses: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    File paths are read and converted to base64 data URLs for images,
    or input_audio format for audio files.

    Args:
        content: Gradio-format content (string or list of content blocks).

    Returns:
        OpenAI-format content.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    openai_content: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            openai_content.append(item)
            continue

        item_type = item.get("type")

        if item_type == "text":
            openai_content.append({"type": "text", "text": item.get("text", "")})

        elif item_type == "file":
            file_info = item.get("file", {})
            file_path = file_info.get("path", "") if isinstance(file_info, dict) else ""

            if not file_path or not os.path.exists(file_path):
                logger.warning("File not found: %s", file_path)
                openai_content.append({"type": "text", "text": f"[File not found: {file_path}]"})
                continue

            # Determine MIME type from file extension
            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type and mime_type.startswith("image/"):
                try:
                    b64_data, actual_mime = get_image(file_path)
                    data_url = f"data:{actual_mime};base64,{b64_data}"
                    openai_content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception as e:
                    logger.warning("Failed to convert image %s: %s", file_path, e)
                    openai_content.append({"type": "text", "text": f"[Image could not be processed: {file_path}]"})

            elif mime_type and mime_type.startswith("audio/"):
                try:
                    b64_data, _ = get_audio(file_path)
                    # Extract format from extension
                    _, ext = os.path.splitext(file_path.lower())
                    audio_format = ext.lstrip(".") or "wav"
                    openai_content.append(
                        {"type": "input_audio", "input_audio": {"data": b64_data, "format": audio_format}}
                    )
                except Exception as e:
                    logger.warning("Failed to convert audio %s: %s", file_path, e)
                    openai_content.append({"type": "text", "text": f"[Audio could not be processed: {file_path}]"})

            else:
                # Unknown file type - try as image first, fall back to text
                try:
                    b64_data, actual_mime = get_image(file_path)
                    data_url = f"data:{actual_mime};base64,{b64_data}"
                    openai_content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception:
                    logger.warning("Unknown file type, could not process: %s", file_path)
                    openai_content.append({"type": "text", "text": f"[File could not be processed: {file_path}]"})

        else:
            # Pass through unknown types
            openai_content.append(item)

    return openai_content


def convert_gradio_messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of Gradio-format messages to OpenAI-format messages.

    Args:
        messages: List of Gradio-format messages.

    Returns:
        List of OpenAI-format messages.
    """
    openai_messages = []
    for msg in messages:
        openai_msg = {"role": msg.get("role", "user")}
        content = msg.get("content")
        if content is not None:
            openai_msg["content"] = convert_gradio_content_to_openai(content)
        openai_messages.append(openai_msg)
    return openai_messages


def convert_openai_content_to_gradio(content: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Any]]:
    """Convert OpenAI-format content to Gradio-format content.

    OpenAI uses: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    Gradio uses: {"type": "file", "file": {"path": "/path/to/image.png"}}

    Base64 data URLs are decoded and saved to temporary files since Gradio
    requires file paths for media content.

    Args:
        content: OpenAI-format content (string or list of content blocks).

    Returns:
        Gradio-format content.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    gradio_content: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            gradio_content.append(item)
            continue

        item_type = item.get("type")

        if item_type == "text":
            gradio_content.append({"type": "text", "text": item.get("text", "")})

        elif item_type == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""

            if url.startswith("data:"):
                try:
                    # Parse data URL: data:image/png;base64,<data>
                    header, b64_data = url.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]  # e.g., "image/png"
                    ext = mime_type.split("/")[1]  # e.g., "png"

                    temp_path = _save_base64_to_temp_file(b64_data, ext)
                    gradio_content.append({"type": "file", "file": {"path": temp_path}})
                except (ValueError, IndexError) as e:
                    logger.warning("Failed to decode base64 image: %s", e)
                    gradio_content.append({"type": "text", "text": "[Image could not be displayed]"})
            else:
                # Regular URL - use as file path
                gradio_content.append({"type": "file", "file": {"path": url}})

        elif item_type == "input_audio":
            audio_data = item.get("input_audio", {})
            if isinstance(audio_data, dict):
                data = audio_data.get("data", "")
                audio_format = audio_data.get("format", "wav")

                try:
                    temp_path = _save_base64_to_temp_file(data, audio_format)
                    gradio_content.append({"type": "file", "file": {"path": temp_path}})
                except (ValueError, IndexError) as e:
                    logger.warning("Failed to decode base64 audio: %s", e)
                    gradio_content.append({"type": "text", "text": "[Audio could not be displayed]"})
            else:
                gradio_content.append(item)

        else:
            # Pass through unknown types
            gradio_content.append(item)

    return gradio_content


def convert_openai_messages_to_gradio(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of OpenAI-format messages to Gradio-format messages.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        List of Gradio-format messages.
    """
    gradio_messages = []
    for msg in messages:
        gradio_msg = {"role": msg.get("role", "user")}
        content = msg.get("content")
        if content is not None:
            gradio_msg["content"] = convert_openai_content_to_gradio(content)
        gradio_messages.append(gradio_msg)
    return gradio_messages


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
# All helpers now include chatbot history as the last element of the returned tuple


def _empty_prompt_response(history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Build response for invalid/unavailable prompt selection."""
    return (
        {"name": None, "arguments": []},
        gr.update(visible=False),
        gr.update(visible=False),
        *hide_all_inputs(),
        history,  # Return chatbot history unchanged
        openai_history,  # Return OpenAI history unchanged
    )


def _empty_resource_response(history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Build response for invalid/unavailable resource selection."""
    return (
        {"name": None, "uri": None, "variables": []},
        gr.update(visible=False),
        gr.update(visible=False),
        *hide_all_inputs(),
        history,  # Return chatbot history unchanged
        openai_history,  # Return OpenAI history unchanged
    )


def _prompt_content_response(
    messages: List[Dict[str, Any]], history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]
) -> Tuple[Any, ...]:
    """Build response showing prompt content and injecting messages into chat history."""
    content = format_messages_for_display(messages) if messages else "No content"
    gradio_messages = convert_openai_messages_to_gradio(messages)
    return (
        {"name": None, "arguments": []},
        gr.update(visible=True, value=content),
        gr.update(visible=False),
        *hide_all_inputs(),
        history + gradio_messages,  # Inject messages into chatbot (Gradio format)
        openai_history + messages,  # Inject messages into state (OpenAI format)
    )


def _resource_content_response(
    messages: List[Dict[str, Any]], history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]
) -> Tuple[Any, ...]:
    """Build response showing resource content and injecting messages into chat history."""
    content = format_messages_for_display(messages) if messages else "No content"
    gradio_messages = convert_openai_messages_to_gradio(messages)
    return (
        {"name": None, "uri": None, "variables": []},
        gr.update(visible=True, value=content),
        gr.update(visible=False),
        *hide_all_inputs(),
        history + gradio_messages,  # Inject messages into chatbot (Gradio format)
        openai_history + messages,  # Inject messages into state (OpenAI format)
    )


def _prompt_input_response(
    prompt_name: str,
    prompt_arguments: List[Any],
    history: List[Dict[str, Any]],
    openai_history: List[Dict[str, Any]],
) -> Tuple[Any, ...]:
    """Build response showing input fields for prompt arguments."""
    return (
        {"name": prompt_name, "arguments": list(prompt_arguments)},
        gr.update(visible=False),
        gr.update(visible=True),
        *build_prompt_input_updates(list(prompt_arguments)),
        history,  # Return chatbot history unchanged (waiting for submit)
        openai_history,  # Return OpenAI history unchanged
    )


def _resource_input_response(
    resource_name: str,
    uri: str,
    variables: List[str],
    history: List[Dict[str, Any]],
    openai_history: List[Dict[str, Any]],
) -> Tuple[Any, ...]:
    """Build response showing input fields for resource variables."""
    return (
        {"name": resource_name, "uri": uri, "variables": variables},
        gr.update(visible=False),
        gr.update(visible=True),
        *build_resource_input_updates(variables),
        history,  # Return chatbot history unchanged (waiting for submit)
        openai_history,  # Return OpenAI history unchanged
    )


def _submit_empty_response(history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """Build response for empty submit (no name in state)."""
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        history,  # Return chatbot history unchanged
        openai_history,  # Return OpenAI history unchanged
    )


def _submit_content_response(
    messages: List[Dict[str, Any]], history: List[Dict[str, Any]], openai_history: List[Dict[str, Any]]
) -> Tuple[Any, ...]:
    """Build response showing submitted content and injecting messages into chat history."""
    content = format_messages_for_display(messages) if messages else "No content"
    gradio_messages = convert_openai_messages_to_gradio(messages)
    return (
        gr.update(visible=True, value=content),
        gr.update(visible=False),
        history + gradio_messages,  # Inject messages into chatbot (Gradio format)
        openai_history + messages,  # Inject messages into state (OpenAI format)
    )


class MCPHandler:
    """Handler for MCP prompt/resource interactions in sync Gradio apps."""

    def __init__(self, chat_factory: ChatFactory):
        """Initialize handler.

        Args:
            chat_factory: ChatFactory instance for instantiating prompts/resources.
        """
        self.chat_factory = chat_factory

    def on_prompt_selected(
        self,
        prompt_name: str,
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Handle prompt selection - show content directly or show input fields.

        Args:
            prompt_name: Name of the selected prompt.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not prompt_name or prompt_name == "No prompts available":
            return _empty_prompt_response(current_history, current_openai_history)

        prompt, prompt_arguments = search_prompt(self.chat_factory.mcp_prompts, prompt_name)
        if not prompt:
            return _empty_prompt_response(current_history, current_openai_history)

        if not prompt_arguments:
            messages = self.chat_factory.instantiate_prompt(prompt_name, lambda _: {})
            return _prompt_content_response(messages or [], current_history, current_openai_history)

        return _prompt_input_response(prompt_name, prompt_arguments, current_history, current_openai_history)

    def on_prompt_submit(
        self,
        state: Dict[str, Any],
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
        *input_values: str,
    ) -> Tuple[Any, ...]:
        """Handle prompt submit - collect values and instantiate prompt.

        Args:
            state: Current prompt state with name and arguments.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).
            *input_values: Values from input textboxes.

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not state.get("name"):
            return _submit_empty_response(current_history, current_openai_history)

        arguments = collect_arguments_from_inputs(state["arguments"], list(input_values))
        messages = self.chat_factory.instantiate_prompt(state["name"], lambda _: arguments)
        return _submit_content_response(messages or [], current_history, current_openai_history)

    def on_resource_selected(
        self,
        resource_name: str,
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Handle resource selection - show content directly or show input fields.

        Args:
            resource_name: Name of the selected resource.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not resource_name or resource_name == "No resources available":
            return _empty_resource_response(current_history, current_openai_history)

        resource, uri, variables = search_resource(self.chat_factory.mcp_resources, resource_name)
        if not resource:
            return _empty_resource_response(current_history, current_openai_history)

        if not variables:
            messages = self.chat_factory.instantiate_resource(resource_name, lambda _: {})
            return _resource_content_response(messages or [], current_history, current_openai_history)

        return _resource_input_response(resource_name, uri, variables, current_history, current_openai_history)  # type: ignore

    def on_resource_submit(
        self,
        state: Dict[str, Any],
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
        *input_values: str,
    ) -> Tuple[Any, ...]:
        """Handle resource submit - collect values and instantiate resource.

        Args:
            state: Current resource state with name, uri, and variables.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).
            *input_values: Values from input textboxes.

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not state.get("name"):
            return _submit_empty_response(current_history, current_openai_history)

        variables = collect_variables_from_inputs(state["variables"], list(input_values))
        messages = self.chat_factory.instantiate_resource(state["name"], lambda _: variables)
        return _submit_content_response(messages or [], current_history, current_openai_history)


class AsyncMCPHandler:
    """Handler for MCP prompt/resource interactions in async Gradio apps."""

    def __init__(self, chat_factory: AsyncChatFactory):
        """Initialize async handler.

        Args:
            chat_factory: AsyncChatFactory instance for instantiating prompts/resources.
        """
        self.chat_factory = chat_factory

    async def on_prompt_selected(
        self,
        prompt_name: str,
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Handle prompt selection - show content directly or show input fields.

        Args:
            prompt_name: Name of the selected prompt.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not prompt_name or prompt_name == "No prompts available":
            return _empty_prompt_response(current_history, current_openai_history)

        prompt, prompt_arguments = search_prompt(self.chat_factory.mcp_prompts, prompt_name)
        if not prompt:
            return _empty_prompt_response(current_history, current_openai_history)

        if not prompt_arguments:
            messages = await self.chat_factory.instantiate_prompt(prompt_name, lambda _: {})
            return _prompt_content_response(messages or [], current_history, current_openai_history)

        return _prompt_input_response(prompt_name, prompt_arguments, current_history, current_openai_history)

    async def on_prompt_submit(
        self,
        state: Dict[str, Any],
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
        *input_values: str,
    ) -> Tuple[Any, ...]:
        """Handle prompt submit - collect values and instantiate prompt.

        Args:
            state: Current prompt state with name and arguments.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).
            *input_values: Values from input textboxes.

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not state.get("name"):
            return _submit_empty_response(current_history, current_openai_history)

        arguments = collect_arguments_from_inputs(state["arguments"], list(input_values))
        messages = await self.chat_factory.instantiate_prompt(state["name"], lambda _: arguments)
        return _submit_content_response(messages or [], current_history, current_openai_history)

    async def on_resource_selected(
        self,
        resource_name: str,
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        """Handle resource selection - show content directly or show input fields.

        Args:
            resource_name: Name of the selected resource.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not resource_name or resource_name == "No resources available":
            return _empty_resource_response(current_history, current_openai_history)

        resource, uri, variables = search_resource(self.chat_factory.mcp_resources, resource_name)
        if not resource:
            return _empty_resource_response(current_history, current_openai_history)

        if not variables:
            messages = await self.chat_factory.instantiate_resource(resource_name, lambda _: {})
            return _resource_content_response(messages or [], current_history, current_openai_history)

        return _resource_input_response(resource_name, uri, variables, current_history, current_openai_history)  # type: ignore

    async def on_resource_submit(
        self,
        state: Dict[str, Any],
        current_history: List[Dict[str, Any]],
        current_openai_history: List[Dict[str, Any]],
        *input_values: str,
    ) -> Tuple[Any, ...]:
        """Handle resource submit - collect values and instantiate resource.

        Args:
            state: Current resource state with name, uri, and variables.
            current_history: Current chatbot history (Gradio format).
            current_openai_history: Current OpenAI-format history (from gr.State).
            *input_values: Values from input textboxes.

        Returns:
            Tuple of updates including updated chatbot and OpenAI histories.
        """
        if not state.get("name"):
            return _submit_empty_response(current_history, current_openai_history)

        variables = collect_variables_from_inputs(state["variables"], list(input_values))
        messages = await self.chat_factory.instantiate_resource(state["name"], lambda _: variables)
        return _submit_content_response(messages or [], current_history, current_openai_history)
