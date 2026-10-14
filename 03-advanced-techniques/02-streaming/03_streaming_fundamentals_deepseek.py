"""DeepSeek Responses API streaming: text deltas and the full event lifecycle."""

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
    "You are a helpful assistant. Keep responses concise and well-structured. "
    "Use markdown formatting when it improves readability."
)


class StreamingChat:
    """Keep local history because DeepSeek's Responses API is stateless."""

    def __init__(self, model: str, tracker: OpenAITokenTracker) -> None:
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.tracker = tracker
        self.messages: list[dict[str, str]] = []

    def stream(self, user_input: str, console: Console, *, show_events: bool = False) -> str:
        """Render text deltas; optionally inspect item boundaries and terminal events."""
        history = [*self.messages, {"role": "user", "content": user_input}]
        final_response: Any = None
        accumulated = ""

        with (
            self.client.responses.create(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=history,
                reasoning={"effort": "none"},
                max_output_tokens=2048,
                stream=True,
            ) as stream,
            Live(Markdown(""), refresh_per_second=15, console=console) as live,
        ):
            for event in stream:
                if event.type == "response.output_text.delta":
                    accumulated += event.delta
                    live.update(Markdown(accumulated))
                elif show_events and event.type in {
                    "response.created",
                    "response.output_item.added",
                    "response.output_item.done",
                    "response.output_text.done",
                }:
                    logger.info("Stream event: %s", event.type)
                elif event.type == "response.completed":
                    final_response = event.response
                elif event.type in {"response.incomplete", "response.failed"}:
                    reason = event.response.error or event.response.incomplete_details
                    raise ValueError(f"DeepSeek stream {event.type}: {reason}")

        if final_response is None:
            raise ValueError("DeepSeek stream ended without a response.completed event")
        if final_response.usage:
            self.tracker.track(final_response.usage)
        answer = final_response.output_text or accumulated
        self.messages = [*history, {"role": "assistant", "content": answer}]
        return answer

    def reset(self) -> None:
        self.messages.clear()


def main() -> None:
    console = Console()
    tracker = OpenAITokenTracker()
    chat = StreamingChat(MODEL, tracker)
    mode = "simple"
    console.print(
        Panel(
            "[bold cyan]DeepSeek Streaming Chat[/bold cyan]\n\n"
            "[green]simple[/green]: render only text deltas\n"
            "[green]events[/green]: also log lifecycle events\n\n"
            "Type [bold]mode simple[/bold], [bold]mode events[/bold], "
            "[bold]clear[/bold], or [bold]quit[/bold].",
            title="02-streaming / 03 — DeepSeek Fundamentals",
        )
    )

    while True:
        console.print(f"\n[dim]Mode: {mode}[/dim]")
        console.print("[bold green]You:[/bold green] ", end="")
        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "clear":
            chat.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if user_input.lower().startswith("mode "):
            requested = user_input.split(" ", 1)[1].strip().lower()
            if requested in {"simple", "events"}:
                mode = requested
            else:
                console.print("[red]Use 'mode simple' or 'mode events'.[/red]")
            continue

        try:
            console.print("\n[bold blue]DeepSeek:[/bold blue]")
            chat.stream(user_input, console, show_events=mode == "events")
        except (OpenAIError, ValueError) as exc:
            logger.error("Streaming failed: %s", exc)
            console.print(f"[red]Error: {exc}[/red]")

    tracker.report()
    console.print(f"[dim]Messages exchanged: {len(chat.messages)}[/dim]")


if __name__ == "__main__":
    main()
