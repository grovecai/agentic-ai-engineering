"""
Tool Use (DeepSeek)

Demonstrates how to enable model to call functions and use tools.
Uses practical tools: calculator, file reader, and bash command execution.
The DeepSeek Responses API is stateless, so this script replays tool calls and results.
"""

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from common import OpenAITokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables from root .env file
load_dotenv(find_dotenv())

# Configure logging
logger = setup_logging(__name__)


# Define available tools in OpenAI-compatible Responses API format
TOOLS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Supports addition, subtraction, multiplication, and division.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Reads the contents of a file at the specified path. Returns the file content as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 100)",
                    "default": 100,
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "run_bash",
        "description": "Runs one of these commands and returns its output: ls, pwd, echo, date.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
]


def calculator(operation: str, a: float, b: float) -> dict[str, Any]:
    """Execute calculator tool."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
    }

    result = operations[operation](a, b)
    logger.info("Calculator: %s %s %s = %s", a, operation, b, result)

    return {"result": result, "operation": operation, "operands": [a, b]}


def read_file(path: str, max_lines: int = 100) -> dict[str, Any]:
    """Read a visible file inside the current lesson directory."""
    try:
        file_path = Path(path).resolve()
        relative_path = file_path.relative_to(Path.cwd().resolve())
        if any(part.startswith(".") for part in relative_path.parts):
            return {"error": "Hidden files are not available to this demo"}
        with file_path.open(encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        content = "".join(lines[:max_lines])
        truncated = total_lines > max_lines

        logger.info("Read file: %s (%d lines)", path, total_lines)

        return {
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "truncated": truncated,
        }
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except ValueError:
        return {"error": "File must be inside the current lesson directory"}
    except OSError as e:
        return {"error": str(e)}


ALLOWED_COMMANDS = {"ls", "pwd", "echo", "date"}


def run_bash(command: str, timeout: int = 30) -> dict[str, Any]:
    """Run an allowed command without invoking a shell."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return {"error": f"Invalid command: {e}"}
    if not argv or argv[0] not in ALLOWED_COMMANDS:
        return {"error": "Command must be one of: ls, pwd, echo, date"}

    logger.info("Running bash command: %s", command)

    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout} seconds"}
    except OSError as e:
        return {"error": str(e)}


# Tool execution mapping
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "read_file": read_file,
    "run_bash": run_bash,
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Execute a tool and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        func = TOOL_FUNCTIONS[tool_name]
        return func(**tool_input)  # type: ignore[operator]
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError) as e:
        logger.error("Tool execution error: %s", e)
        return {"error": str(e)}


class ToolUseChat:
    """Chat session with tool use capabilities."""

    def __init__(
        self,
        model: str,
        token_tracker: OpenAITokenTracker,
        console: Console,
    ):
        """Initialize the chat session with tools."""
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.token_tracker = token_tracker
        self.console = console
        self.messages: list[Any] = []
        self.model = model

    def send_message(self, user_message: str) -> str:
        """Send a message and handle potential tool use."""
        # DeepSeek does not retain server-side context. Replay all prior items.
        messages = [*self.messages, {"role": "user", "content": user_message}]

        # Bound the loop in case the model repeatedly asks for more tools.
        for _ in range(10):
            logger.info("API call (history items: %d)", len(messages))

            # Make API call with tools using Responses API
            response = self.client.responses.create(
                model=self.model,
                max_output_tokens=4096,
                tools=TOOLS,
                input=messages,
                reasoning={"effort": "none"},
            )

            # Track tokens
            if response.usage:
                self.token_tracker.track(response.usage)

            # Replay the model's message and any function calls on the next request.
            messages.extend(response.output)

            # Check if the response contains function calls
            function_calls = [o for o in response.output if o.type == "function_call"]

            if function_calls:
                # Execute tools
                self.console.print("\n[yellow]-> Executing tools...[/yellow]")

                for func_call in function_calls:
                    function_name = func_call.name
                    try:
                        function_args = json.loads(func_call.arguments)
                    except json.JSONDecodeError:
                        function_args = None

                    self.console.print(
                        f"  [dim]* {function_name}({json.dumps(function_args, indent=2)})[/dim]"
                    )

                    # Execute the tool
                    result = (
                        execute_tool(function_name, function_args)
                        if isinstance(function_args, dict)
                        else {"error": "Tool arguments must be a JSON object"}
                    )

                    # Add function call output to messages for next iteration
                    messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": func_call.call_id,
                            "output": json.dumps(result),
                        }
                    )

                # Continue the loop to get final response
                continue

            else:
                # No function calls, extract and return text response
                answer = response.output_text or ""
                if not answer:
                    raise ValueError("DeepSeek returned no text response")
                self.messages = messages
                return answer

        raise RuntimeError("Tool-use loop exceeded 10 model calls")

    def get_message_count(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.messages)


def main() -> None:
    """Main orchestration function that handles user interaction and coordinates the chat flow."""
    console = Console()
    token_tracker = OpenAITokenTracker()
    chat = ToolUseChat(os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), token_tracker, console)

    # Welcome message
    console.print(
        Panel(
            "[bold cyan]DeepSeek Agent with Tools![/bold cyan]\n\n"
            "Available tools:\n"
            "* Calculator (add, subtract, multiply, divide)\n"
            "* Read file (read contents of any file)\n"
            "* Run a simple command (ls, pwd, echo, date)\n\n"
            "Try: 'What's 123 * 456?' or 'List files in the current directory'\n"
            "Or: 'Read the pyproject.toml file'\n\n"
            "Type 'quit' to exit.",
            title="Tool Use Demo",
        )
    )

    # Chat loop
    try:
        while True:
            console.print("\n[bold green]You:[/bold green] ", end="")
            user_input = input().strip()

            if user_input.lower() in ["quit", "exit", ""]:
                console.print("\n[yellow]Ending chat session...[/yellow]")
                break

            try:
                response = chat.send_message(user_input)

                if response:
                    console.print("\n[bold blue]Agent:[/bold blue]")
                    console.print(Markdown(response))

            except (OpenAIError, ValueError, RuntimeError) as e:
                logger.error("Error during chat: %s", e)
                console.print(f"\n[red]Error: {e}[/red]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Ending chat session...[/yellow]")

    # Report usage
    console.print()
    token_tracker.report()
    console.print(f"\n[dim]Total messages exchanged: {chat.get_message_count()}[/dim]")


if __name__ == "__main__":
    main()
