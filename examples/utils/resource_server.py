import logging

from mcp.server.fastmcp import FastMCP
from mcp_multi_server.utils import configure_logging


logger = logging.getLogger(__name__)

mcp = FastMCP("Resource Server")


@mcp._mcp_server.set_logging_level()
async def set_logging_level(level: str) -> None:
    configure_logging(name="resource_server", level=level)
    logger.info(f"Resource server logging level set to {level}")


if __name__ == "__main__":
    logger.info("Starting MCP Resource Server...")
    mcp.run()
