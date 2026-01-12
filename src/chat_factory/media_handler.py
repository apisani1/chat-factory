"""Media handling utilities for displaying and processing various content types."""

import base64
import io
import tempfile
import urllib.error
import urllib.request

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
)
from PIL import Image as PilImage
from PIL.Image import (
    Image,
    Resampling,
)


def get_image(image_path: str) -> tuple[str, str]:
    """
    Load an image from the given path, resize it to a maximum of 1024x1024 pixels,
    convert it to PNG format, and return its base64-encoded data along with the MIME type.
    """
    img: Image = PilImage.open(image_path)
    img.thumbnail((1024, 1024))  # Resize to max 1024x1024

    # Convert to RGB to ensure consistent format
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Save as PNG to a bytes buffer in memory
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"


def get_audio(audio_path: str) -> tuple[str, str]:
    """ "
    Load an audio file from the given path and return its base64-encoded data along with the MIME type.
    """
    import os

    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # Get file extension and determine MIME type
    _, ext = os.path.splitext(audio_path.lower())
    ext = ext.lstrip(".")

    # Map common audio extensions to MIME types
    audio_mime_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
    }

    mime_type = audio_mime_types.get(ext, "audio/mpeg")  # Default to mp3
    return base64.b64encode(audio_data).decode("utf-8"), mime_type


def open_file_with_system_default(file_path: str) -> None:
    """Open a file with the system's default application."""
    import os
    import platform
    import subprocess

    system = platform.system()
    if system == "Darwin":  # macOS
        subprocess.run(["open", file_path], check=True)
    elif system == "Windows":
        os.startfile(file_path)  # type: ignore # pylint: disable=no-member
    elif system == "Linux":
        subprocess.run(["xdg-open", file_path], check=True)
    else:
        raise OSError(f"Unable to open file on {system}")


def display_image_content(image_content: ImageContent) -> None:
    """Display an image from ImageContent by decoding base64 data and showing it."""
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_content.data)

        # Assumes that the server sends PNG data
        image: Image = PilImage.open(io.BytesIO(image_bytes))

        # Scale up the image so it's visible
        image = image.resize((256, 256), Resampling.NEAREST)

        # Display image (this will open in default image viewer)
        image.show()

        print(f"  Displayed image: {image.size} pixels, MIME type: {image_content.mimeType}")

    except Exception as e:
        print(f"  Error displaying image: {e}")


def play_audio_content(audio_content: AudioContent) -> None:
    """Display/play an audio from AudioContent by decoding base64 data and playing it."""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_content.data)

        # Determine file extension from MIME type
        mime_to_ext = {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
            "audio/mp4": ".m4a",
            "audio/aac": ".aac",
        }

        file_ext = mime_to_ext.get(audio_content.mimeType, ".mp3")

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        # Try to play the audio using the system's default audio player
        try:
            open_file_with_system_default(temp_file_path)
            print(f"  Playing audio: {len(audio_bytes)} bytes, MIME type: {audio_content.mimeType}")
            print(f"  Temporary file: {temp_file_path}")

        except Exception as play_error:
            print(f"  Error playing audio: {play_error}")
            print(f"  Audio saved to: {temp_file_path}")

    except Exception as e:
        print(f"  Error playing audio: {e}")


def display_pdf_content(pdf_bytes: bytes, uri: str) -> None:
    """Display PDF content by saving to a temporary file and opening it."""
    try:
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        # Try to open the PDF using the system's default PDF viewer
        try:
            open_file_with_system_default(temp_file_path)
            print(f"  PDF opened: {len(pdf_bytes)} bytes from {uri}")
            print(f"  Temporary file: {temp_file_path}")

        except Exception as open_error:
            print(f"  Error opening PDF: {open_error}")
            print(f"  PDF saved to: {temp_file_path}")

    except Exception as e:
        print(f"  Error displaying PDF: {e}")


def display_html_content(html_bytes: bytes, uri: str) -> None:
    """Display HTML content by saving to a temporary file and opening it in browser."""
    try:
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="wb") as temp_file:
            temp_file.write(html_bytes)
            temp_file_path = temp_file.name

        # Try to open the HTML file in the default browser
        try:
            open_file_with_system_default(temp_file_path)
            print(f"  HTML opened in browser: {len(html_bytes)} bytes from {uri}")
            print(f"  Temporary file: {temp_file_path}")

        except Exception as open_error:
            print(f"  Error opening HTML: {open_error}")
            print(f"  HTML saved to: {temp_file_path}")

    except Exception as e:
        print(f"  Error displaying HTML: {e}")


def decode_binary_file(embedded_resource: EmbeddedResource, output_path: str) -> None:
    """Decode binary file from EmbeddedResource and save it to disk."""
    try:
        if isinstance(embedded_resource.resource, BlobResourceContents):
            # Decode base64 data
            binary_data = base64.b64decode(embedded_resource.resource.blob)

            # Write to file
            with open(output_path, "wb") as output_file:
                output_file.write(binary_data)

            print(f"  Binary file decoded and saved to: {output_path}")
            print(f"  File size: {len(binary_data)} bytes")
        else:
            print(f"  Unsupported resource type: {type(embedded_resource.resource)}")

    except Exception as e:
        print(f"  Error decoding binary file: {e}")


def load_content_from_uri(resource_link: ResourceLink) -> bytes:
    """Load content from a URI and return the raw bytes."""
    try:
        with urllib.request.urlopen(str(resource_link.uri)) as response:
            content = response.read()
            return content
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Failed to load content from URI {resource_link.uri}: {e}")


def display_content_from_uri(resource_link: ResourceLink) -> None:
    """Display content from a URI based on its MIME type."""
    try:
        # Load content from URI
        content_bytes = load_content_from_uri(resource_link)
        mime_type = resource_link.mimeType or "application/octet-stream"

        # Handle different content types based on MIME type
        if mime_type.startswith("image/"):
            # Convert to ImageContent and display
            image_content = ImageContent(
                type="image", data=base64.b64encode(content_bytes).decode("utf-8"), mimeType=mime_type
            )
            display_image_content(image_content)
        elif mime_type.startswith("audio/"):
            # Convert to AudioContent and display
            audio_content = AudioContent(
                type="audio", data=base64.b64encode(content_bytes).decode("utf-8"), mimeType=mime_type
            )
            play_audio_content(audio_content)
        elif mime_type == "application/pdf":
            display_pdf_content(content_bytes, str(resource_link.uri))
        elif mime_type in ["text/html", "application/xhtml+xml"]:
            display_html_content(content_bytes, str(resource_link.uri))
        else:
            # For other content types, offer to save to file
            print(f"  Content loaded: {len(content_bytes)} bytes")
            filename = input(f"Enter filename to save {mime_type} content to (or press Enter to skip): ").strip()
            if filename:
                with open(filename, "wb") as f:
                    f.write(content_bytes)
                print(f"  Content saved to: {filename}")
            else:
                print("  Content not saved")

    except Exception as e:
        print(f"  Error loading content from URI: {e}")
