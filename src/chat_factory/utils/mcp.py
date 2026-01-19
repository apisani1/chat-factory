from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from mcp.types import (
    AudioContent,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    Prompt,
    PromptArgument,
    Resource,
    ResourceLink,
    ResourceTemplate,
    TextContent,
)
from mcp_multi_server.utils import extract_template_variables


def process_tool_result_content(tool_result: CallToolResult) -> str:
    """Process tool result content blocks and convert to OpenAI tool response format.

    Args:
        tool_result: CallToolResult from MCP server.

    Returns:
        String content for OpenAI tool response (images and audio converted to text descriptions).
    """

    def convert_mcp_content_to_tool_response(
        content_block: ContentBlock,
    ) -> Dict[str, Any]:
        """Convert MCP content block to OpenAI tool message format.

        Tool messages must always be text-only (no images/audio arrays).
        Images and audio are converted to text descriptions.

        Args:
            content_block: Content block from MCP tool result.

        Returns:
            Dict with 'type' and 'text' keys, suitable for OpenAI tool messages.
        """
        if isinstance(content_block, TextContent):
            return {"type": "text", "text": content_block.text}
        if isinstance(content_block, ImageContent):
            return {"type": "text", "text": f"[Image: {content_block.mimeType} received]"}
        if isinstance(content_block, AudioContent):
            return {"type": "text", "text": f"[Audio: {content_block.mimeType} received]"}
        if isinstance(content_block, EmbeddedResource):
            if hasattr(content_block.resource, "text"):
                return {"type": "text", "text": content_block.resource.text}  # type: ignore
            return {"type": "text", "text": "[Embedded resource: binary data received]"}
        if isinstance(content_block, ResourceLink):
            return {"type": "text", "text": f"[Resource link: {content_block.uri}]"}
        return {"type": "text", "text": "[Unknown content type received]"}

    text_parts = []
    for content_block in tool_result.content:
        # Convert to OpenAI tool format (always returns dict with 'text' key)
        converted = convert_mcp_content_to_tool_response(content_block)
        text_parts.append(converted["text"])

    # Join all parts into a single string (required for tool role messages)
    return "\n".join(text_parts) if text_parts else ""


def search_prompt(
    prompts: Dict[str, Prompt], prompt_name: str
) -> Tuple[Optional[Prompt], Optional[List[PromptArgument]]]:
    """Search for a prompt by name in the given prompts dictionary.

    Args:
        prompts: Dictionary of available prompts.
        prompt_name: Name of the prompt to search for.

    Returns:
        The Prompt object and its arguments if found, else None.
    """
    prompt = prompts.get(prompt_name)
    if prompt:
        prompt_arguments = prompt.arguments
        return prompt, prompt_arguments  # type: ignore
    return None, None


def convert_mcp_content_to_message(
    content_block: ContentBlock,
) -> Union[str, List[Dict[str, Any]]]:
    """Convert MCP content block to OpenAI user/assistant message format.

    Returns plain string for text content, array for media content (images/audio).
    This format is suitable for user and assistant messages, which can include
    rich media content that OpenAI's vision API can process.

    Args:
        content_block: Content block from MCP prompt or resource.

    Returns:
        String for text-only content, array list for media content.
    """
    if isinstance(content_block, TextContent):
        return content_block.text
    if isinstance(content_block, ImageContent):
        # Return array with image_url for OpenAI vision API
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_block.mimeType};base64,{content_block.data}"},
            }
        ]
    if isinstance(content_block, AudioContent):
        # Standard GPT-4 cannot process audio, inform the LLM it was played locally
        return [
            {
                "type": "text",
                "text": f"[Audio content ({content_block.mimeType}) was played locally for the user but cannot be processed by the AI]",
            }
        ]
    if isinstance(content_block, EmbeddedResource):
        if hasattr(content_block.resource, "text"):
            return content_block.resource.text  # type: ignore
        # Pending: handle other embedded resource types appropriately
        content_block_text = str(content_block.resource)
        return f"[Embedded resource: {content_block_text[:min(80, len(content_block_text))]}]"
    if isinstance(content_block, ResourceLink):
        return f"[Resource link: {content_block.uri}]"
    return "[Unknown content type received]"


def search_resource(
    resources: Dict[str, Union[Resource, ResourceTemplate]], resource_name: str
) -> Tuple[Optional[Resource], Optional[str], Optional[List[str]]]:
    """Search for a prompt by name in the given prompts dictionary.

    Args:
        resources: Dictionary of available resources.
        resource_name: Name of the resource to search for.

    Returns:
        The Resource object, its uri and variables if found, else None.
    """
    resource = resources.get(resource_name)
    uri = None
    variables = None
    if resource:
        if hasattr(resource, "uri"):
            uri = resource.uri  # type: ignore[union-attr]
        elif hasattr(resource, "uriTemplate"):
            uri = resource.uriTemplate  # type: ignore[union-attr]
            variables = extract_template_variables(uri)
        else:
            resource = None
    return resource, uri, variables  # type: ignore
