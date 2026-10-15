"""Discover an MCP tool, bridge it into DeepSeek, and answer one question."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness import Agent, Tool, client_from_env


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=[str(Path(__file__).with_name("server.py"))]
    )
    async with stdio_client(params) as (read, write):  # noqa: SIM117
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = (await session.list_tools()).tools
            print("MCP tools:", [tool.name for tool in discovered])
            # The sync model loop runs in a worker thread; calls cross back to this async MCP session.
            loop = asyncio.get_running_loop()

            def execute(args: dict) -> str:
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool("lookup_document", arguments=args), loop
                )
                result = future.result(timeout=10)
                if result.isError:
                    raise ValueError("MCP tool returned an error")
                return "\n".join(block.text for block in result.content if hasattr(block, "text"))

            tools = [
                Tool(t.name, t.description or "", t.inputSchema, execute)
                for t in discovered
                if t.name == "lookup_document"
            ]
            if not tools:
                raise RuntimeError("MCP lookup tool was not discovered")
            answer = await asyncio.to_thread(
                Agent(client_from_env(), tools).run, "When should we roll back a deployment?"
            )
            print(answer)


if __name__ == "__main__":
    asyncio.run(main())
