import base64
import mimetypes

from mcp.server.fastmcp import FastMCP
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
)
from mcp_multi_server.utils import configure_logging
from chat_factory.media_handler import (
    get_audio,
    get_image,
)


# Create server
mcp = FastMCP("MCP Tool Server")


@mcp._mcp_server.set_logging_level()
async def set_logging_level(level: str) -> None:
    configure_logging(name="mcp", level=level)


# Tools returning non text types for tool demonstration purposes
# Currently OpenAI function calling only supports text-based outputs and the
# chat client example will just display the media content from the tool call
# and send a text message to the LLM acknowledging the media received.


@mcp.tool(name="get_image")
def get_image_tool(image_path: str) -> CallToolResult:
    """Load an image file and return its contents as base64-encoded image content.

    Reads image files from the filesystem and returns them in MCP ImageContent format
    with automatic MIME type detection. Supports common image formats (PNG, JPEG, GIF, etc.).

    Parameters:
        image_path (str): Absolute or relative file path to the image file

    Returns:
        CallToolResult containing:
        - isError: False
        - content: List with single ImageContent object containing:
          - type: "image"
          - data: Base64-encoded image data
          - mimeType: Detected MIME type (e.g., "image/png", "image/jpeg")

    Raises:
        FileNotFoundError if image file doesn't exist
        IOError if file cannot be read

    Example:
        result = get_image("/path/to/product-photo.png")
        # Returns ImageContent with base64 data and mimeType="image/png"

    Note:
        - MIME type is auto-detected from file extension
        - Image data is base64-encoded for safe transmission
        - Used for displaying product images or visual content
    """
    image_data, mime_type = get_image(image_path)
    return CallToolResult(isError=False, content=[ImageContent(type="image", data=image_data, mimeType=mime_type)])


@mcp.tool(name="get_audio")
def get_audio_tool(audio_path: str) -> CallToolResult:
    """Load an audio file and return its contents as base64-encoded audio content.

    Reads audio files from the filesystem and returns them in MCP AudioContent format
    with automatic MIME type detection. Supports common audio formats (MP3, WAV, OGG, etc.).

    Parameters:
        audio_path (str): Absolute or relative file path to the audio file

    Returns:
        CallToolResult containing:
        - isError: False
        - content: List with single AudioContent object containing:
          - type: "audio"
          - data: Base64-encoded audio data
          - mimeType: Detected MIME type (e.g., "audio/mpeg", "audio/wav")

    Raises:
        FileNotFoundError if audio file doesn't exist
        IOError if file cannot be read

    Example:
        result = get_audio("/path/to/product-demo.mp3")
        # Returns AudioContent with base64 data and mimeType="audio/mpeg"

    Note:
        - MIME type is auto-detected from file extension
        - Audio data is base64-encoded for safe transmission
        - Used for instructions or audio content
    """
    audio_data, mime_type = get_audio(audio_path)
    return CallToolResult(isError=False, content=[AudioContent(type="audio", data=audio_data, mimeType=mime_type)])


@mcp.tool(name="get_file")
def get_file_tool(file_path: str) -> CallToolResult:
    """Load any file and return its contents as an embedded resource with base64 encoding.

    Reads any file type from the filesystem and returns it as an MCP EmbeddedResource
    with automatic MIME type detection. Use this for general file access (documents,
    data files, etc.) when specific image/audio tools don't apply.

    Parameters:
        file_path (str): Absolute or relative file path to any file

    Returns:
        CallToolResult containing:
        - isError: False
        - content: List with single EmbeddedResource object containing:
          - type: "resource"
          - resource: BlobResourceContents with:
            - uri: File URI (e.g., "file:///path/to/file.pdf")
            - blob: Base64-encoded file contents
            - mimeType: Auto-detected MIME type or "application/octet-stream" if unknown

    Raises:
        FileNotFoundError if file doesn't exist
        IOError if file cannot be read

    Example:
        result = get_file("/path/to/manual.pdf")
        # Returns EmbeddedResource with base64 blob and mimeType="application/pdf"

    Note:
        - Handles any file type (PDFs, spreadsheets, text files, binary files, etc.)
        - MIME type auto-detected from file extension
        - Falls back to "application/octet-stream" for unknown types
        - File contents are base64-encoded for safe transmission
        - For images use get_image, for audio use get_audio (they provide optimized formats)
    """
    with open(file_path, "rb") as file:
        file_data = file.read()
    encoded = base64.b64encode(file_data).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)
    return CallToolResult(
        isError=False,
        content=[
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=f"file://{file_path}",  # type: ignore[arg-type]
                    blob=encoded,
                    mimeType=mime_type or "application/octet-stream",
                ),
            )
        ],
    )


@mcp.tool(name="get_uri_content")
def get_uri_content_tool(content_uri: str) -> CallToolResult:
    """Create a resource link for a content URI without loading the uri contents.

    Returns a ResourceLink reference to remote or local content via URI. Unlike get_file,
    this doesn't load the file contents—it just creates a link reference. Useful for
    referencing remote URLs, large files, or streaming content.

    Parameters:
        content_uri (str): URI to the content (URL, file://, or other URI scheme)

    Returns:
        CallToolResult containing:
        - isError: False
        - content: List with single ResourceLink object containing:
          - type: "resource_link"
          - name: Extracted filename from URI (last path segment)
          - uri: The provided content URI
          - mimeType: Auto-detected from URI extension or "application/octet-stream"

    Example:
        result = get_content_uri("https://example.com/manual.pdf")
        # Returns ResourceLink(name="manual.pdf", uri="https://...", mimeType="application/pdf")

        result = get_content_uri("file:///local/video.mp4")
        # Returns ResourceLink(name="video.mp4", uri="file://...", mimeType="video/mp4")

    Note:
        - Does NOT load file contents (unlike get_file)
        - Creates a reference/link to content
        - MIME type inferred from URI extension
        - Supports any URI scheme (http://, https://, file://, etc.)
        - Useful for remote resources, streaming, or avoiding large file transfers
        - Name is extracted from the last segment of the URI path
    """
    mime_type, _ = mimetypes.guess_type(content_uri)
    return CallToolResult(
        isError=False,
        content=[
            ResourceLink(
                type="resource_link",
                name=content_uri.split("/")[-1],
                uri=content_uri,  # type: ignore[arg-type]
                mimeType=mime_type or "application/octet-stream",
            )
        ],
    )


if __name__ == "__main__":
    print("Starting MCP Tool Server...")
    mcp.run()
