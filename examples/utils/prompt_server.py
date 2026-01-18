import base64
import logging
import mimetypes
from typing import List

from media_handler import (
    get_audio,
    get_image,
)
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import (
    AssistantMessage,
    Message,
    UserMessage,
)
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from mcp_multi_server.utils import configure_logging


logger = logging.getLogger(__name__)

mcp = FastMCP("Prompt Server")


@mcp._mcp_server.set_logging_level()
async def set_logging_level(level: str) -> None:
    configure_logging(name="prompt_server", level=level)
    logger.info(f"Prompt server logging level set to {level}")


@mcp.prompt()
def inventory_check() -> str:
    """Creates a prompt that returns a text-based response.
    The prompt requests a list of products in a given category along with their inventory status.
    Args:
        category: the product category to check
    """

    prompt = """
    Consult the inventory database and list every product that needs restocking, providing its name, SKU,
    on-hand quantity, and supplier.
    """

    return prompt


@mcp.prompt()
def category_promotion(category: str, discount_percentage: str) -> str:
    """Creates a prompt that returns a text-based response.
    The prompt requests a list of products in a given category with updated prices
    after applying a specified discount percentage.
    Args:
        category: the product category to promote
        discount_percentage: the discount percentage to offer
    """

    prompt = """
    Find all inventory items for products in the {category} category.
    Update the prices of all of the above inventory item by reducing them by a {discount_percentage}%.
    List the updated products names and their new prices.
    """

    return prompt.format(
        category=category,
        discount_percentage=discount_percentage,
    )


@mcp.prompt()
def inventory_restock_brief(category: str, min_stock: int) -> List[Message]:
    """Generates a multi-message prompt.
    The prompt requests a list of products in a given category and recommended restock amounts for low inventory items.
    Args:
        category: the product category to check
        min_stock: the minimum stock level to flag for restocking
    """
    return [
        UserMessage(content=f"Consult the inventory database and list every product in the {category} category."),
        AssistantMessage(content="Only show the information requested in the next message."),
        UserMessage(
            content=TextContent(
                type="text",
                text=(
                    f"""For each product returned, checks the inventory items associated and display the folling
                    information: name, SKU, on-hand quantity, and supplier.
                    Flag entries where quantity is below {min_stock} units and recommend a restock amount."""
                ),
            ),
        ),
    ]


@mcp.prompt()
def load_image(file_path: str) -> List[Message]:
    """Loads an image file and returns its contents as an image content."""
    image_data, image_mime_type = get_image(file_path)
    return [UserMessage(content=ImageContent(type="image", data=image_data, mimeType=image_mime_type))]


@mcp.prompt()
def load_audio(file_path: str) -> List[Message]:
    """Loads an audio file and returns its contents as an audio content."""
    audio_data, audio_mime_type = get_audio(file_path)
    return [UserMessage(content=AudioContent(type="audio", data=audio_data, mimeType=audio_mime_type))]


@mcp.prompt()
def load_file(file_path: str) -> List[Message]:
    """Loads a file and returns its contents as an embedded resource."""
    with open(file_path, "rb") as file:
        file_data = file.read()
    encoded = base64.b64encode(file_data).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)
    return [
        UserMessage(
            content=EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=f"file://{file_path}",  # type: ignore[arg-type]
                    blob=encoded,
                    mimeType=mime_type or "application/octet-stream",
                ),
            )
        )
    ]


@mcp.prompt()
def load_uri_content(content_uri: str) -> List[Message]:
    """Sends a content URI as an resource link."""
    mime_type, _ = mimetypes.guess_type(content_uri)
    return [
        UserMessage(
            content=ResourceLink(
                type="resource_link",
                name=content_uri.split("/")[-1],
                uri=content_uri,  # type: ignore[arg-type]
                mimeType=mime_type or "application/octet-stream",
            )
        )
    ]


if __name__ == "__main__":
    logger.info("Starting MCP Prompt Server...")
    mcp.run()
