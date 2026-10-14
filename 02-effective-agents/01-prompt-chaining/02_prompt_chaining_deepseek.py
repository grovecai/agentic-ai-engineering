"""Prompt chaining with DeepSeek: topic -> outline -> draft -> edited post."""

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common import OpenAITokenTracker, interactive_menu, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

OUTPUT_DIR = Path("output")
MODEL = os.getenv("DEEPSEEK_CHAIN_MODEL", "deepseek-v4-pro")
WRITER_MODEL = os.getenv("DEEPSEEK_WRITER_MODEL", "deepseek-flash")

SUGGESTED_TOPICS = [
    "Practical Async Programming in Python",
    "AI Agents in Production",
    "WebAssembly Beyond the Browser",
    "Zero-Trust Security for Startups",
]

OUTLINER_SYSTEM_PROMPT = (
    "You are a research planner. Given a topic, identify 3-5 broad areas that cover "
    "distinct dimensions of the subject. Output the topic title on the first line, "
    "then the areas as short bullet points. No extra commentary."
)
WRITER_SYSTEM_PROMPT = (
    "You are a technical blog writer. Given an outline, write a concise blog post. "
    "Use the title as an H1 heading and each bullet point as an H2 section. "
    "Write 1-2 short paragraphs per section, under 1000 words total. "
    "You have no web search access: do not claim to have checked current sources, "
    "invent citations, or present time-sensitive claims as verified facts."
)
EDITOR_SYSTEM_PROMPT = (
    "You are a professional editor. Polish the given blog post for grammar, clarity, "
    "and flow. Preserve its factual scope. At the end, add a '## Key Takeaways' "
    "section with 3-5 bullet points. Return the complete edited post."
)

ChainCallback = Callable[[str, dict[str, Any]], None]


class PromptChain:
    """Run three focused DeepSeek calls in a fixed sequence."""

    def __init__(self, model: str, writer_model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.writer_model = writer_model
        self.token_tracker = token_tracker

    def _call_llm(self, model: str, system: str, prompt: str) -> str:
        logger.info("Calling %s", model)
        response = self.client.responses.create(
            model=model,
            max_output_tokens=8192,
            instructions=system,
            input=prompt,
            reasoning={"effort": "none"},
        )
        if response.usage:
            self.token_tracker.track(response.usage)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise ValueError(
                f"DeepSeek response was {response.status} ({reason or 'unknown reason'})"
            )
        result = (response.output_text or "").strip()
        if not result:
            raise ValueError("DeepSeek produced empty output; aborting chain.")
        return result

    def run(self, topic: str, on_event: ChainCallback | None = None) -> str:
        """Pass each stage's text to the next stage."""
        notify = on_event or (lambda _event, _data: None)

        notify("step_start", {"name": "Outline"})
        outline = self._call_llm(
            self.model, OUTLINER_SYSTEM_PROMPT, f"Create a blog outline for: {topic}"
        )
        self.token_tracker.report()
        notify("step_complete", {"name": "Outline", "result": outline})

        notify("step_start", {"name": "Write"})
        draft = self._call_llm(
            self.writer_model,
            WRITER_SYSTEM_PROMPT,
            f"Write a full blog post from this outline:\n\n{outline}",
        )
        self.token_tracker.report()
        notify("step_complete", {"name": "Write"})

        notify("step_start", {"name": "Edit"})
        final = self._call_llm(
            self.model, EDITOR_SYSTEM_PROMPT, f"Edit and polish this blog post:\n\n{draft}"
        )
        self.token_tracker.report()
        notify("step_complete", {"name": "Edit"})
        notify("chain_complete", {})
        return final


def main() -> None:
    console = Console()
    token_tracker = OpenAITokenTracker()

    def on_chain_event(event: str, data: dict[str, Any]) -> None:
        if event == "step_start":
            console.print(f"  [cyan]{data['name']}...[/cyan]")
        elif event == "step_complete":
            console.print("  [green]✓[/green] Done")
            if data["name"] == "Outline":
                console.print(Panel(data["result"], title="Outline", border_style="dim"))

    header = Panel(
        "[bold cyan]Prompt Chaining — DeepSeek[/bold cyan]\n\n"
        "Topic → [Outliner] → [Writer] → [Editor] → Final Post\n\n"
        "Each step feeds its output to the next.",
        title="Prompt Chaining",
    )

    try:
        while True:
            topic = interactive_menu(
                console,
                SUGGESTED_TOPICS,
                title="Select a Topic",
                header=header,
                allow_custom=True,
                custom_prompt="Enter your topic",
            )
            if not topic:
                break

            try:
                chain = PromptChain(MODEL, WRITER_MODEL, token_tracker)
                result = chain.run(topic, on_event=on_chain_event)
                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = re.sub(r"[^\w]+", "_", topic.lower()).strip("_")[:50] or "article"
                path = OUTPUT_DIR / f"{slug}.md"
                path.write_text(result, encoding="utf-8")

                console.print("\n[bold blue]Final Article:[/bold blue]")
                console.print(Markdown(result))
                console.print(f"\n[dim]Saved to {path.resolve()}[/dim]")
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            except (OpenAIError, ValueError, OSError, KeyError, EOFError) as exc:
                logger.error("Chain failed: %s", exc)
                console.print(f"\n[red]Error: {exc}[/red]")
            finally:
                token_tracker.reset()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
