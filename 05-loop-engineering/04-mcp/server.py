"""A local MCP server exposing the same lookup document."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server.fastmcp import FastMCP

from harness import lookup

mcp = FastMCP("local-operations-docs")


@mcp.tool()
def lookup_document(doc_id: str) -> str:
    """Read the local runbook or incident document by ID."""
    return lookup({"doc_id": doc_id})


if __name__ == "__main__":
    mcp.run(transport="stdio")
