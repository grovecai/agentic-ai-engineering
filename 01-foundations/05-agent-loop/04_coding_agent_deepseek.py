"""
Agent Loop (DeepSeek)

Demonstrates a minimal autonomous agent that:
- Takes a task from the user
- Decides which tools to use
- Executes tools in a loop until complete
"""

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from common.logging_config import setup_logging
from common.token_tracking import OpenAITokenTracker
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables from root .env file
load_dotenv(find_dotenv())

# Configure logging
logger = setup_logging(__name__)

SYSTEM_PROMPT = """You are a coding agent. Use the provided tools to complete tasks.

Guidelines:
- Read files before modifying them
- Make changes incrementally and verify each step
- If a command fails, analyze the error and try a different approach
- The bash tool supports only ls and pwd; use read_file for file contents
- When done, provide a brief summary of what you accomplished"""


# Tool definitions (OpenAI-compatible Responses API format)
TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read",
                }
            },
            "additionalProperties": False,
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write",
                },
            },
            "additionalProperties": False,
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "bash",
        "description": "Run ls or pwd and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "additionalProperties": False,
            "required": ["command"],
        },
    },
]


def _lesson_path(path: str) -> Path:
    """Resolve a visible path inside the current lesson directory."""
    resolved = Path(path).resolve()
    relative = resolved.relative_to(Path.cwd().resolve())
    if any(part.startswith(".") for part in relative.parts):
        raise ValueError("Hidden files are not available to this demo")
    return resolved


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    if name == "read_file":
        try:
            return _lesson_path(tool_input["path"]).read_text(encoding="utf-8")
        except (KeyError, TypeError, ValueError, OSError) as e:
            return f"Error: {e}"

    elif name == "write_file":
        try:
            _lesson_path(tool_input["path"]).write_text(tool_input["content"], encoding="utf-8")
            return f"Successfully wrote to {tool_input['path']}"
        except (KeyError, TypeError, ValueError, OSError) as e:
            return f"Error: {e}"

    elif name == "bash":
        try:
            argv = shlex.split(tool_input["command"])
            if not argv or argv[0] not in {"ls", "pwd"}:
                return "Error: bash tool supports only ls and pwd"
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            output = result.stdout + result.stderr
            return output if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except (KeyError, TypeError, ValueError, OSError) as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


class CodingAgent:
    """
    Minimal autonomous coding agent.

    Executes tools in a loop until the task is complete.
    """

    def __init__(self, model: str | None = None):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.max_iterations = 10
        self.token_tracker = OpenAITokenTracker()

    def run(self, task: str) -> str:
        """Execute the agent loop for the given task."""
        logger.info(f"Task: {task}")

        input_messages: list[Any] = [{"role": "user", "content": task}]
        for iteration in range(self.max_iterations):
            logger.info(f"--- Iteration {iteration + 1} ---")

            # Call the model using responses API
            response = self.client.responses.create(
                model=self.model,
                tools=TOOLS,
                instructions=SYSTEM_PROMPT,
                input=input_messages,
                reasoning={"effort": "none"},
            )

            if response.usage:
                self.token_tracker.track(response.usage)

            # Log any text output
            if response.output_text:
                logger.info(f"🤖 Agent: {response.output_text}")

            # Check if there are function calls
            function_calls = [o for o in response.output if o.type == "function_call"]

            # If no function calls, task is complete
            if not function_calls:
                return response.output_text or "Done"

            # DeepSeek is stateless: preserve model outputs alongside tool results.
            input_messages.extend(response.output)

            # Execute tools and collect results
            tool_outputs: list[dict[str, str]] = []
            for call in function_calls:
                try:
                    args = json.loads(call.arguments)
                except json.JSONDecodeError as e:
                    args = None
                    logger.error("Invalid tool arguments: %s", e)

                logger.info(f"🔧 Tool: {call.name}({json.dumps(args)})")
                result = (
                    execute_tool(call.name, args)
                    if isinstance(args, dict)
                    else "Error: Tool arguments must be a JSON object"
                )
                logger.info(f"📋 Result: {result[:100]}{'...' if len(result) > 100 else ''}")

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps({"result": result}),
                    }
                )

            # Replay the full conversation on the next API request.
            input_messages.extend(tool_outputs)

        return "Max iterations reached"


def main() -> None:
    """Main orchestration function."""
    console = Console()
    console.print(
        Panel(
            "Examples:\n"
            "  - Create a calculator following the style of existing files\n"
            "  - List current dependencies\n"
            "  - Explain code in the current folder\n\n"
            "Type 'quit' to exit.",
            title="Coding Agent (DeepSeek)",
        )
    )

    agent = CodingAgent()

    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            user_input = input().strip()

            if user_input.lower() in ("exit", "quit", "q", ""):
                console.print("\n[yellow]Ending session...[/yellow]")
                break

            try:
                response = agent.run(user_input)
                console.print("\n[bold blue]Agent:[/bold blue]")
                console.print(Markdown(response))
            except (OpenAIError, ValueError, RuntimeError) as e:
                console.print(f"\n[red]Error: {e}[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")

    console.print()
    agent.token_tracker.report()


if __name__ == "__main__":
    main()
