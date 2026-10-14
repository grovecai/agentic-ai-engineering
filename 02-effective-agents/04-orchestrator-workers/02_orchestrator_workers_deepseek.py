"""Plan research with DeepSeek, run workers in parallel, and synthesize an article."""

import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
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
MODEL = os.getenv("DEEPSEEK_ORCHESTRATOR_MODEL", "deepseek-v4-pro")
WORKER_MODEL = os.getenv("DEEPSEEK_WORKER_MODEL", "deepseek-flash")

SUGGESTED_TOPICS = [
    "Compare Bun vs Node.js for Backend Development",
    "Python 3.13 New Features and Performance",
    "WebAssembly in Production: State of the Art",
    "Compare Relational and Document Databases",
]

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are a research orchestrator. Break the topic into 2-4 specific, independent "
    "subtopics covering distinct dimensions. Give each worker a focused question. "
    "Also provide synthesis_instructions explaining how the final writer should connect "
    "the worker findings into one article. "
    "Always call create_research_plan."
)
WORKER_SYSTEM_PROMPT = (
    "You are a technical analyst. Answer the assigned question in 2-3 substantive "
    "paragraphs with useful examples and clear uncertainty. You have no web search "
    "access: do not invent citations, benchmarks, release details, or claim that "
    "current facts have been verified."
)
SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a senior technical writer. Combine the supplied worker analyses into "
    "one coherent article with an H1 title and clear H2 sections. Avoid repetition. "
    "Use only the supplied analyses; do not invent evidence or citations. "
    "If the analyses leave a current fact uncertain, say so."
)

PLAN_TOOL = {
    "type": "function",
    "name": "create_research_plan",
    "description": "Break a topic into independent questions for parallel workers.",
    "parameters": {
        "type": "object",
        "properties": {
            "subtopics": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short subtopic title"},
                        "research_prompt": {
                            "type": "string",
                            "description": "Specific question for the worker",
                        },
                    },
                    "required": ["title", "research_prompt"],
                    "additionalProperties": False,
                },
            },
            "synthesis_instructions": {
                "type": "string",
                "description": "How to connect the worker findings in the final article",
            },
        },
        "required": ["subtopics", "synthesis_instructions"],
        "additionalProperties": False,
    },
}

OrchestratorCallback = Callable[[str, dict[str, Any]], None]


class OrchestratorWorkers:
    """Let an LLM plan independent work, then synthesize its parallel results."""

    def __init__(self, model: str, worker_model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.worker_model = worker_model
        self.token_tracker = token_tracker
        self._usage_lock = Lock()
        self._notify: OrchestratorCallback = lambda _event, _data: None

    def _request(
        self, *, model: str, system: str, prompt: str, max_output_tokens: int, **kwargs: Any
    ) -> Any:
        logger.info("Calling %s", model)
        response = self.client.responses.create(
            model=model,
            instructions=system,
            input=prompt,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": "none"},
            **kwargs,
        )
        if response.usage:
            with self._usage_lock:
                self.token_tracker.track(response.usage)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise ValueError(
                f"DeepSeek response was {response.status} ({reason or 'unknown reason'})"
            )
        return response

    def _call_text(self, *, model: str, system: str, prompt: str, max_output_tokens: int) -> str:
        response = self._request(
            model=model, system=system, prompt=prompt, max_output_tokens=max_output_tokens
        )
        result = (response.output_text or "").strip()
        if not result:
            raise ValueError("DeepSeek produced empty text.")
        return result

    def _plan(self, topic: str) -> dict[str, Any]:
        response = self._request(
            model=self.model,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            prompt=f"Plan independent research for: {topic}",
            max_output_tokens=1024,
            tools=[PLAN_TOOL],
            tool_choice={"type": "function", "name": "create_research_plan"},
        )
        calls = [
            item
            for item in response.output
            if item.type == "function_call" and item.name == "create_research_plan"
        ]
        if len(calls) != 1:
            raise ValueError("Orchestrator did not return exactly one research plan.")
        try:
            plan = json.loads(calls[0].arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Orchestrator returned invalid JSON.") from exc
        if not isinstance(plan, dict):
            raise TypeError("Orchestrator returned a non-object plan.")
        subtopics = plan.get("subtopics")
        instructions = plan.get("synthesis_instructions")
        if not isinstance(subtopics, list) or not 2 <= len(subtopics) <= 4:
            raise ValueError("Research plan must contain 2-4 subtopics.")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("Research plan has no synthesis instructions.")
        titles: set[str] = set()
        for subtopic in subtopics:
            if not isinstance(subtopic, dict):
                raise TypeError("Research subtopic must be an object.")
            title = subtopic.get("title")
            question = subtopic.get("research_prompt")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("Research subtopic has no title.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Research subtopic has no question.")
            if title in titles:
                raise ValueError(f"Duplicate research subtopic: {title}")
            titles.add(title)
        return plan

    def _research_subtopic(self, subtopic: dict[str, str]) -> dict[str, str]:
        title = subtopic["title"]
        logger.info("Worker analyzing: %s", title)
        content = self._call_text(
            model=self.worker_model,
            system=WORKER_SYSTEM_PROMPT,
            prompt=f"Subtopic: {title}\n\nQuestion: {subtopic['research_prompt']}",
            max_output_tokens=2048,
        )
        return {"title": title, "content": content}

    def _synthesize(self, topic: str, research: list[dict[str, str]], instructions: str) -> str:
        sections = "\n\n---\n\n".join(f"## {r['title']}\n\n{r['content']}" for r in research)
        return self._call_text(
            model=self.model,
            system=f"{SYNTHESIZER_SYSTEM_PROMPT}\n\nPlan instructions: {instructions}",
            prompt=f"Topic: {topic}\n\nWorker analyses:\n\n{sections}\n\nWrite the complete article.",
            max_output_tokens=8192,
        )

    def run(self, topic: str, on_event: OrchestratorCallback | None = None) -> str:
        self._notify = on_event or (lambda _event, _data: None)
        self._notify("plan_start", {})
        plan = self._plan(topic)
        subtopics = plan["subtopics"]
        logger.info("Synthesis instructions: %s", plan["synthesis_instructions"])
        self.token_tracker.report()
        self._notify("plan_complete", {"subtopics": subtopics})

        self._notify("workers_start", {"count": len(subtopics)})
        research_by_index: dict[int, dict[str, str]] = {}
        failed_titles: list[str] = []
        with ThreadPoolExecutor(max_workers=len(subtopics)) as executor:
            futures = {
                executor.submit(self._research_subtopic, subtopic): index
                for index, subtopic in enumerate(subtopics)
            }
            for future in as_completed(futures):
                index = futures[future]
                title = subtopics[index]["title"]
                try:
                    research_by_index[index] = future.result()
                    self._notify("worker_complete", {"title": title})
                except (OpenAIError, ValueError) as exc:
                    logger.error("Worker failed on %s: %s", title, exc)
                    failed_titles.append(title)
                    self._notify("worker_failed", {"title": title, "error": str(exc)})
        self.token_tracker.report()
        if failed_titles:
            raise RuntimeError(f"Worker research failed: {', '.join(failed_titles)}")

        research = [research_by_index[index] for index in range(len(subtopics))]
        self._notify("synthesize_start", {})
        final = self._synthesize(topic, research, plan["synthesis_instructions"])
        self.token_tracker.report()
        self._notify("synthesize_complete", {})
        return final


def main() -> None:
    console = Console()
    token_tracker = OpenAITokenTracker()

    def on_event(event: str, data: dict[str, Any]) -> None:
        if event == "plan_start":
            console.print("\n[bold yellow]Orchestrator:[/bold yellow] Planning research...")
        elif event == "plan_complete":
            subtopics = data["subtopics"]
            console.print(
                Panel(
                    "\n".join(f"• {item['title']}" for item in subtopics),
                    title=f"Research Plan ({len(subtopics)} subtopics)",
                    border_style="cyan",
                )
            )
        elif event == "workers_start":
            console.print(
                f"\n[bold yellow]Workers:[/bold yellow] "
                f"Analyzing {data['count']} subtopics in parallel..."
            )
        elif event == "worker_complete":
            console.print(f"  [green]✓[/green] {data['title']}")
        elif event == "worker_failed":
            console.print(f"  [red]✗[/red] {data['title']}: {data['error']}")
        elif event == "synthesize_start":
            console.print("\n[bold yellow]Synthesizer:[/bold yellow] Combining analyses...")
        elif event == "synthesize_complete":
            console.print("  [green]✓[/green] Done")

    header = Panel(
        "[bold cyan]Orchestrator-Workers — DeepSeek[/bold cyan]\n\n"
        "Topic → [Orchestrator] → Dynamic subtopics\n"
        "      → [Worker 1] + [Worker 2] + [Worker N] (parallel)\n"
        "      → [Synthesizer] → Final Article",
        title="Orchestrator-Workers",
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
                orchestrator = OrchestratorWorkers(MODEL, WORKER_MODEL, token_tracker)
                result = orchestrator.run(topic, on_event=on_event)
                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = re.sub(r"[^\w]+", "_", topic.lower()).strip("_")[:50] or "article"
                path = OUTPUT_DIR / f"{slug}.md"
                path.write_text(result, encoding="utf-8")

                console.print("\n[bold blue]Final Article:[/bold blue]")
                console.print(Markdown(result))
                console.print(f"\n[dim]Saved to {path.resolve()}[/dim]")
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            except (
                OpenAIError,
                ValueError,
                TypeError,
                RuntimeError,
                OSError,
                KeyError,
                EOFError,
            ) as exc:
                logger.error("Orchestration failed: %s", exc)
                console.print(f"\n[red]Error: {exc}[/red]")
            finally:
                token_tracker.reset()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
