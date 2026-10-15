"""Compose skills, hooks, sandbox, MCP, child agents, and compaction."""

import asyncio
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from harness import Agent, Hooks, Tool, client_from_env
from skills import SkillRegistry

run_snippet = runpy.run_path(str(ROOT / "03-sandboxing/demo.py"))["run_snippet"]
research = runpy.run_path(str(ROOT / "05-subagents/demo.py"))["research"]
compact = runpy.run_path(str(ROOT / "06-context-compaction/demo.py"))["compact"]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "04-mcp/server.py")])
    async with stdio_client(params) as (read, write):  # noqa: SIM117
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = (await session.list_tools()).tools
            mcp_schema = next(t for t in discovered if t.name == "lookup_document")
            loop = asyncio.get_running_loop()

            def mcp_lookup(args: dict) -> str:
                future = asyncio.run_coroutine_threadsafe(
                    session.call_tool("lookup_document", arguments=args), loop
                )
                result = future.result(timeout=10)
                if result.isError:
                    raise ValueError("MCP lookup failed")
                return "\n".join(block.text for block in result.content if hasattr(block, "text"))

            registry = SkillRegistry(ROOT / "01-skills/skills")
            tools = [
                Tool(
                    mcp_schema.name,
                    mcp_schema.description or "",
                    mcp_schema.inputSchema,
                    mcp_lookup,
                ),
                *registry.tools(),
                Tool(
                    "delegate_research",
                    "Delegate up to two independent questions to isolated child agents.",
                    {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 2,
                            }
                        },
                        "required": ["questions"],
                    },
                    research,
                ),
                Tool(
                    "run_fixed_calculation",
                    "Run a fixed 2+3+5 check in a limited subprocess; takes no arguments.",
                    {"type": "object", "properties": {}},
                    lambda _: run_snippet("print(sum([2, 3, 5]))"),
                ),
            ]
            allowed = {tool.name for tool in tools}

            def approve(name: str, args: dict) -> None:
                if name not in allowed:
                    raise PermissionError(f"Tool {name} is not enabled")
                print(f"tool: {name} {args}")

            agent = Agent(client_from_env(), tools, Hooks(before=[approve]), max_steps=7)
            answer = await asyncio.to_thread(
                agent.run,
                "Summarize incident response with its template and deployment rollback. "
                "Delegate one question for each topic, check both local documents through MCP, "
                "and run the fixed calculation as a tool demonstration.",
                registry.catalog_prompt()
                + "\nUse tools for facts. Cite document IDs. Keep the final answer brief.",
            )
            print(answer)
            print("Skills loaded on demand:", sorted(registry.loaded))
            print("Parent token usage:", agent.last_usage)
            first_turn = agent.last_messages
            follow_up = await asyncio.to_thread(
                agent.run,
                "Give a one-sentence takeaway from the documents.",
                history=first_turn,
            )
            print("Follow-up:", follow_up)
            reduced = await asyncio.to_thread(compact, agent.last_messages, client_from_env())
            print(f"Compacted history: {len(agent.last_messages)} -> {len(reduced)} messages")


if __name__ == "__main__":
    asyncio.run(main())
