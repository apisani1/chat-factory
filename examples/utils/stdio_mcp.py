from typing import List

from utils.media_handler import (
    decode_binary_file,
    display_content_from_uri,
    display_image_content,
    play_audio_content,
)
from mcp.types import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptArgument,
    ResourceLink,
    TextContent,
)


def display_mcp_content(content_block: ContentBlock) -> None:
    """Display a content block to the user based on its type.

    Args:
        content_block: Content block from MCP tool result or prompt.
    """
    if isinstance(content_block, TextContent):
        print(f"[Result] {content_block.text}\n")
    elif isinstance(content_block, ImageContent):
        print("[Result] Image content received")
        display_image_content(content_block)
    elif isinstance(content_block, AudioContent):
        print(f"[Result] Audio content received ({content_block.mimeType})")
        play_audio_content(content_block)
    elif isinstance(content_block, EmbeddedResource):
        if hasattr(content_block.resource, "text"):
            print(f"[Result] Embedded resource text: {content_block.resource.text}\n")  # type: ignore[attr-defined]
        else:
            print("[Result] Embedded resource blob")
            filename = input("Enter filename to save embedded resource (or press Enter to skip): ").strip()
            if filename:
                decode_binary_file(content_block, filename)
    elif isinstance(content_block, ResourceLink):
        print(f"[Result] Resource link: {content_block.uri}")
        display_content_from_uri(content_block)
    else:
        # Unknown content type
        content_block_text = str(content_block)
        print(f"[Result] {content_block_text[:min(80, len(content_block_text))]}\n")


def display_mcp_resource_result(resource_result: str) -> None:
    print(f"[Result] {resource_result}\n")


def get_prompt_arguments(prompt_arguments: List[PromptArgument]) -> dict[str, str]:
    """Ask user for prompt arguments interactively."""
    if not prompt_arguments:
        return {}
    print("Please provide values for the prompt following arguments:")
    arguments: dict[str, str] = {}
    for arg in prompt_arguments:
        required_text = "(required)" if arg.required else "(optional)"
        user_input = input(f"Enter {arg.name} {required_text}: ").strip()
        if user_input or arg.required:
            arguments[arg.name] = user_input
    return arguments


def get_template_variables(variables: List[str]) -> dict[str, str]:
    """Ask user for values for variables extracted from URI template"""
    print("Please provide values for the following template variables:")
    values = {}
    for var in variables:
        value = input(f"Enter value for {var}: ").strip()
        values[var] = value
    return values
