"""DeepSeek streaming agent: show deltas, execute completed tool calls, resume."""

import ast
import json
import operator
import os
from typing import Any

from common import OpenAITokenTracker, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = os.getenv("DEEPSEEK_STREAMING_MODEL", "deepseek-flash")
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools for weather and calculations. "
    "Weather data is a fixed demo snapshot, not a live forecast. "
    "After using tools, explain the results concisely."
)
TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get a fixed demo weather snapshot for Tokyo, London, or Paris.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]
WEATHER_DATA = {
    "tokyo": {"temp_c": 14, "conditions": "Clear"},
    "london": {"temp_c": 9, "conditions": "Overcast"},
    "paris": {"temp_c": 11, "conditions": "Light rain"},
}
OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_calculate_node(node.left), _calculate_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_calculate_node(node.operand))
    raise ValueError("Only numbers, parentheses, +, -, *, and / are supported")


def execute_tool(name: str, arguments: str) -> str:
    """Validate completed JSON arguments before calling a local tool."""
    try:
        data = json.loads(arguments)
        if not isinstance(data, dict):
            raise TypeError("Tool arguments must be a JSON object")
        if name == "get_weather":
            city = data.get("city")
            if not isinstance(city, str):
                raise ValueError("city must be a string")
            snapshot = WEATHER_DATA.get(city.strip().lower())
            if snapshot is None:
                raise ValueError("Demo weather is available only for Tokyo, London, and Paris")
            result = {"city": city, "demo_snapshot": True, **snapshot}
        elif name == "calculate":
            expression = data.get("expression")
            if not isinstance(expression, str) or len(expression) > 100:
                raise ValueError("expression must be a string of at most 100 characters")
            result = {
                "expression": expression,
                "result": _calculate_node(ast.parse(expression, mode="eval")),
            }
        else:
            raise ValueError(f"Unknown tool: {name}")
        return json.dumps(result)
    except (json.JSONDecodeError, ValueError, TypeError, ZeroDivisionError, SyntaxError) as exc:
        return json.dumps({"error": str(exc)})


class StreamingAgent:
    def __init__(self, model: str, tracker: OpenAITokenTracker) -> None:
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.tracker = tracker
        self.messages: list[dict[str, Any]] = []

    def _stream_response(self, console: Console) -> Any:
        """Display text and tool indicators; return the completed response."""
        completed: Any = None
        text = ""
        with (
            self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=self.messages,
                tools=TOOLS,
                reasoning={"effort": "none"},
                max_output_tokens=2048,
                stream=True,
            ) as stream,
            Live(Markdown(""), refresh_per_second=15, console=console) as live,
        ):
            for event in stream:
                if event.type == "response.output_text.delta":
                    text += event.delta
                    live.update(Markdown(text))
                elif event.type == "response.output_item.added":
                    if event.item.type == "function_call":
                        console.print(f"[dim]Calling {event.item.name}...[/dim]")
                elif event.type == "response.function_call_arguments.delta":
                    logger.debug("Tool argument fragment: %s", event.delta)
                elif event.type == "response.completed":
                    completed = event.response
                elif event.type in {"response.incomplete", "response.failed"}:
                    reason = event.response.error or event.response.incomplete_details
                    raise ValueError(f"DeepSeek stream {event.type}: {reason}")

        if completed is None:
            raise ValueError("DeepSeek stream ended without a response.completed event")
        if completed.usage:
            self.tracker.track(completed.usage)
        return completed

    def run(self, user_input: str, console: Console) -> str:
        """Stream one turn, execute all tool calls, and stream the continuation."""
        self.messages.append({"role": "user", "content": user_input})
        full_text = ""
        for _ in range(8):
            response = self._stream_response(console)
            calls = []
            for item in response.output:
                if item.type == "message":
                    content = "".join(
                        part.text for part in item.content if part.type == "output_text"
                    )
                    if content:
                        full_text += ("\n\n" if full_text else "") + content
                        self.messages.append({"role": "assistant", "content": content})
                elif item.type == "function_call":
                    calls.append(item)
                    self.messages.append(
                        {
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

            if not calls:
                return full_text

            for call in calls:
                output = execute_tool(call.name, call.arguments)
                console.print(f"[dim]  {call.name} → {output}[/dim]")
                self.messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )
        raise RuntimeError("Agent exceeded eight tool-call rounds")

    def reset(self) -> None:
        self.messages.clear()


def main() -> None:
    console = Console()
    tracker = OpenAITokenTracker()
    agent = StreamingAgent(MODEL, tracker)
    console.print(
        Panel(
            "[bold cyan]DeepSeek Streaming Agent[/bold cyan]\n\n"
            "Watch text deltas and tool calls as they arrive. Completed tool calls "
            "run after each response, then the next response streams.\n\n"
            "Try: [bold]Compare the demo weather in Tokyo and London, then calculate "
            "the temperature difference.[/bold]\n\n"
            "Type [bold]clear[/bold] or [bold]quit[/bold].",
            title="02-streaming / 04 — DeepSeek Agent",
        )
    )
    while True:
        console.print("\n[bold green]You:[/bold green] ", end="")
        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "clear":
            agent.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        try:
            console.print("\n[bold blue]DeepSeek:[/bold blue]")
            agent.run(user_input, console)
        except (OpenAIError, ValueError, RuntimeError) as exc:
            logger.error("Agent failed: %s", exc)
            console.print(f"[red]Error: {exc}[/red]")
    tracker.report()


if __name__ == "__main__":
    main()
