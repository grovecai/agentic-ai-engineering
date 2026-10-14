"""Codebase Navigator using DeepSeek's stateless Responses API.

The repository index, search tools, and persistent memory are shared with the
Anthropic example. Only the model API and tool-use loop differ.
"""

import json
import os
from typing import Any

from common import setup_logging
from common.menu import interactive_menu
from common.token_tracking import OpenAITokenTracker
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from indexer.embedder import Embedder
from store.memory import MemoryStore
from store.vector import VectorStore
from tools.files import FILE_TOOLS, execute_list_directory, execute_read_file
from tools.memory import MEMORY_TOOLS, execute_recall_memory, execute_save_memory
from tools.repo import REPO_TOOLS, execute_clone_and_index, execute_list_repos
from tools.search import SEARCH_TOOLS, execute_grep, execute_search_code

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

SUGGESTED_REPOS = [
    "openai/swarm",
    "strands-agents/sdk-python",
    "anthropics/anthropic-sdk-python",
    "pallets/flask",
]

SYSTEM_PROMPT = """You are a Codebase Navigator helping engineers explore codebases.

When a user names a GitHub repository, use clone_and_index before searching it.
Use search_code for conceptual questions, grep for exact identifiers or patterns,
and read_file when search results do not provide enough context. Include file
paths and line numbers in answers when possible.

Save useful architectural facts, insights, and user preferences with save_memory.
Use recall_memory when you need earlier context. Be concise and explain how the
relevant code works."""


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Convert the shared Anthropic-style schemas to Responses API tools."""
    definitions = MEMORY_TOOLS + REPO_TOOLS + FILE_TOOLS + SEARCH_TOOLS
    return [
        {
            "type": "function",
            "name": definition["name"],
            "description": definition["description"],
            "parameters": definition["input_schema"],
        }
        for definition in definitions
    ]


class DeepSeekCodeNavigatorAgent:
    """Run the navigator's tools through DeepSeek's Responses API."""

    def __init__(self, model: str | None = None) -> None:
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.token_tracker = OpenAITokenTracker()
        self.tools = _build_tool_definitions()
        self.messages: list[Any] = []
        self.max_iterations = 15

        self.memory = MemoryStore()
        self.vector_store = VectorStore()
        self.embedder = Embedder()

    def _build_instructions(self) -> str:
        memory_summary = self.memory.summary()
        if memory_summary != "No memories stored yet.":
            return SYSTEM_PROMPT + f"\n\n## Recalled Memories\n{memory_summary}"
        return SYSTEM_PROMPT

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        dispatch = {
            "save_memory": lambda inp: execute_save_memory(self.memory, inp),
            "recall_memory": lambda inp: execute_recall_memory(self.memory, inp),
            "clone_and_index": lambda inp: execute_clone_and_index(
                self.vector_store, self.embedder, inp
            ),
            "list_repos": lambda inp: execute_list_repos(self.vector_store, inp),
            "read_file": lambda inp: execute_read_file(self.vector_store, inp),
            "list_directory": lambda inp: execute_list_directory(self.vector_store, inp),
            "search_code": lambda inp: execute_search_code(self.vector_store, self.embedder, inp),
            "grep": lambda inp: execute_grep(self.vector_store, self.embedder, inp),
        }
        handler = dispatch.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        try:
            return str(handler(tool_input))
        except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model
            logger.error("Tool '%s' failed: %s", name, exc)
            return f"Error executing {name}: {exc}"

    def chat(self, user_message: str, console: Console) -> str:
        """Replay conversation, execute function calls, and return a final answer."""
        messages = [*self.messages, {"role": "user", "content": user_message}]

        for _ in range(self.max_iterations):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    max_output_tokens=4096,
                    instructions=self._build_instructions(),
                    tools=self.tools,
                    input=messages,
                    reasoning={"effort": "none"},
                )
            except RateLimitError as exc:
                logger.warning("Rate limited: %s", exc)
                return f"Rate limited: {exc}"
            except OpenAIError as exc:
                logger.error("API error: %s", exc)
                return f"API error: {exc}"

            if response.usage:
                self.token_tracker.track(response.usage)

            # DeepSeek does not keep server-side conversation state. Replay both
            # model output items and the corresponding tool results next time.
            messages.extend(response.output)
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                self.messages = messages
                return response.output_text or "Done."

            for call in function_calls:
                input_summary = str(call.arguments)
                if len(input_summary) > 160:
                    input_summary = input_summary[:157] + "..."
                logger.info("Calling tool %s with %s", call.name, input_summary)
                console.print(f"  [dim][tool: {call.name}] {input_summary}[/dim]")

                try:
                    tool_input = json.loads(call.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    tool_input = None
                    result = f"Invalid tool arguments: {exc}"
                else:
                    if isinstance(tool_input, dict):
                        result = self._execute_tool(call.name, tool_input)
                    else:
                        result = "Tool arguments must be a JSON object"

                result_preview = result.splitlines()[0][:80] if result else "(empty result)"
                logger.info("Tool %s returned: %s", call.name, result_preview)
                console.print(f"  [dim]  → {result_preview}[/dim]")
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

        self.messages = messages
        return "Reached maximum iterations. Please try a more specific question."


def main() -> None:
    agent = DeepSeekCodeNavigatorAgent()
    console = Console()
    header = Panel(
        "An LLM enhanced with retrieval, tools, and memory.\n\n"
        "Select a repo to index, then ask questions about the codebase.",
        title="Codebase Navigator (DeepSeek)",
    )
    repo = interactive_menu(
        console,
        SUGGESTED_REPOS,
        title="Select a Repo to Explore",
        header=header,
        allow_custom=True,
        custom_prompt="Enter owner/repo (e.g., pallets/flask)",
    )
    if not repo:
        return

    console.print(f"\n[bold green]Indexing:[/bold green] {repo}")
    response = agent.chat(f"index the repo {repo}", console)
    console.print("\n[bold blue]Navigator:[/bold blue]")
    console.print(Markdown(response))
    console.print("\n[dim]Ask questions about the codebase. Type 'quit' to exit.[/dim]")

    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            try:
                user_input = input().strip()
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit", "q", ""):
                break
            response = agent.chat(user_input, console)
            console.print("\n[bold blue]Navigator:[/bold blue]")
            console.print(Markdown(response))
            agent.token_tracker.report()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
