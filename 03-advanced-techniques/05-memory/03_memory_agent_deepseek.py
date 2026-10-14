"""Personal assistant with persistent memory using DeepSeek's Responses API."""

import json
import os
import time
from typing import Any

from common import OpenAITokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from memory import MemoryManager

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
SYSTEM_PROMPT = """\
You are a personal assistant with persistent memory. You remember information about the user
across sessions using a three-tier memory system:

1. Working memory — temporary session notes (auto-cleared)
2. Episodic memory — timestamped events and interactions (persisted to JSON)
3. Semantic memory — facts, preferences, and knowledge (persisted to a vector database)

Memory guidelines:
- Store personal facts and preferences in semantic memory.
- Store notable events and interactions in episodic memory.
- Use recall proactively before asking the user for information they may have shared.
- Use importance 0.3-0.5 for routine details, 0.6-0.8 for personal details,
  and 0.9-1.0 for critical information.
- Tell the user when you recall something.

{memory_context}"""

MEMORY_TOOLS = [
    {
        "type": "function",
        "name": "remember",
        "description": "Store information in working, episodic, or semantic memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Information to remember"},
                "memory_type": {
                    "type": "string",
                    "enum": ["working", "episodic", "semantic"],
                },
                "importance": {"type": "number", "description": "Importance from 0 to 1"},
            },
            "required": ["content", "memory_type"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "recall",
        "description": "Search persisted memories for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "limit": {"type": "integer", "description": "Maximum results (default 5)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "forget",
        "description": "Delete a persisted memory by ID and type.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "ID of the memory to delete"},
                "memory_type": {"type": "string", "enum": ["episodic", "semantic"]},
            },
            "required": ["memory_id", "memory_type"],
            "additionalProperties": False,
        },
    },
]


class MemoryAgent:
    """Run the memory tools through DeepSeek's stateless tool-use loop."""

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.token_tracker = OpenAITokenTracker()
        self.memory = MemoryManager()
        self.messages: list[Any] = []
        self.transcript: list[dict[str, str]] = []
        self.max_iterations = 10

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        try:
            if name == "remember":
                return self.memory.remember(
                    content=tool_input["content"],
                    memory_type=tool_input["memory_type"],
                    importance=tool_input.get("importance", 0.5),
                )
            if name == "recall":
                return self.memory.recall(
                    query=tool_input["query"],
                    limit=tool_input.get("limit", 5),
                )
            if name == "forget":
                return self.memory.forget(
                    memory_id=tool_input["memory_id"],
                    memory_type=tool_input["memory_type"],
                )
            return f"Unknown tool: {name}"
        except (KeyError, TypeError, ValueError, OSError) as exc:
            logger.error("Tool '%s' failed: %s", name, exc)
            return f"Error executing {name}: {exc}"

    def chat(self, user_message: str, console: Console) -> str:
        """Replay conversation, execute tool calls, and return the final reply."""
        self.messages.append({"role": "user", "content": user_message})
        self.transcript.append({"role": "user", "content": user_message})

        for _ in range(self.max_iterations):
            try:
                response = self.client.responses.create(
                    model=MODEL,
                    max_output_tokens=4096,
                    instructions=SYSTEM_PROMPT.format(
                        memory_context=self.memory.build_memory_context()
                    ),
                    tools=MEMORY_TOOLS,
                    input=self.messages,
                    reasoning={"effort": "none"},
                )
            except RateLimitError:
                logger.warning("Rate limited — waiting 30s before retry...")
                time.sleep(30)
                continue
            except OpenAIError as exc:
                logger.error("API error: %s", exc)
                return f"API error: {exc}"

            if response.usage:
                self.token_tracker.track(response.usage)
            if response.status != "completed":
                return f"DeepSeek response was {response.status}."

            # DeepSeek does not retain server-side context; keep model outputs and
            # function-call outputs in the input for the next request.
            self.messages.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            if response.output_text:
                self.transcript.append({"role": "assistant", "content": response.output_text})
            if not calls:
                return response.output_text or "Done."

            for call in calls:
                try:
                    args = json.loads(call.arguments)
                    if not isinstance(args, dict):
                        raise TypeError("Tool arguments must be a JSON object")
                    result = self._execute_tool(call.name, args)
                except (json.JSONDecodeError, TypeError) as exc:
                    result = f"Invalid tool arguments: {exc}"

                preview = call.arguments[:80]
                console.print(f"  [dim][tool: {call.name}] {preview}[/dim]")
                console.print(f"  [dim]  → {result.splitlines()[0][:80]}[/dim]")
                self.messages.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": result}
                )

        return "Reached maximum iterations."

    def consolidate(self) -> list[str]:
        """Extract long-term memories at session end with one DeepSeek call."""
        prompt = self.memory.build_consolidation_prompt(self.transcript)
        if not prompt:
            return []
        try:
            response = self.client.responses.create(
                model=MODEL,
                max_output_tokens=1024,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": "none"},
            )
            if response.usage:
                self.token_tracker.track(response.usage)
            if response.status != "completed" or not response.output_text:
                logger.error("Consolidation response was %s or empty", response.status)
                return []
            return self.memory.save_consolidation(response.output_text)
        except OpenAIError as exc:
            logger.error("Consolidation failed: %s", exc)
            return []


def main() -> None:
    console = Console()
    agent = MemoryAgent()
    stats = agent.memory.get_stats()
    loaded = stats["episodic"]["count"] + stats["semantic"]["count"]
    console.print(
        Panel(
            "A personal assistant with working, episodic, and semantic memory.\n"
            f"Loaded {loaded} memories from previous sessions.\n\n"
            'Try: "I prefer Python" or "What do you remember about me?"\n'
            'Type "exit" or "quit" to end the session.',
            title="Memory Systems — DeepSeek",
        )
    )

    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit", "q", ""):
                break
            reply = agent.chat(user_input, console)
            console.print("\n[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(reply))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")

    console.print("\n[dim]Consolidating session memories...[/dim]")
    saved = agent.consolidate()
    if saved:
        console.print(f"[green]Saved {len(saved)} memories for next session:[/green]")
        for item in saved:
            console.print(f"  [dim]{item}[/dim]")
    else:
        console.print("[dim]No new memories to consolidate.[/dim]")
    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
