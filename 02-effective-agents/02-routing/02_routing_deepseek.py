"""Route a topic through one of three DeepSeek content generation chains."""

import json
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
MODEL = os.getenv("DEEPSEEK_ROUTING_MODEL", "deepseek-v4-pro")
LIGHT_MODEL = os.getenv("DEEPSEEK_ROUTING_LIGHT_MODEL", "deepseek-flash")

SUGGESTED_TOPICS = [
    "How to deploy a FastAPI app on AWS Lambda",
    "A new Python release and its impact on developers",
    "What is retrieval-augmented generation (RAG)",
    "How to set up CI/CD with GitHub Actions",
]

CLASSIFY_TOOL = {
    "type": "function",
    "name": "classify_content",
    "description": "Classify a topic by the structure its article needs.",
    "parameters": {
        "type": "object",
        "properties": {
            "content_type": {
                "type": "string",
                "enum": ["tutorial", "news", "concept"],
                "description": "tutorial: how-to; news: an announcement or change; concept: an explanation.",
            },
            "reasoning": {"type": "string", "description": "Brief reason for this choice."},
        },
        "required": ["content_type", "reasoning"],
        "additionalProperties": False,
    },
}

CLASSIFY_SYSTEM_PROMPT = (
    "Classify the topic as tutorial, news, or concept. "
    "Always call classify_content with your decision."
)
NO_SEARCH_GUIDANCE = (
    "You have no web search access. Do not invent citations, versions, release dates, "
    "or claim that time-sensitive facts have been verified."
)

TUTORIAL_PREREQS_PROMPT = (
    "You are a technical writer. List prerequisites for this tutorial as bullets. "
    "Be specific where confident and avoid guessing versions. " + NO_SEARCH_GUIDANCE
)
TUTORIAL_STEPS_PROMPT = (
    "You are a technical writer. Given a topic and prerequisites, write a numbered "
    "step-by-step guide with code examples where relevant. " + NO_SEARCH_GUIDANCE
)
TUTORIAL_TROUBLESHOOTING_PROMPT = (
    "You are a technical support writer. Given a tutorial, add 3-5 common issues "
    "and solutions. Format each as '### Problem' and '**Solution**'."
)
NEWS_SUMMARY_PROMPT = (
    "You are a tech journalist. Summarize changes supplied in the topic as bullets. "
    "If the topic does not provide concrete changes, say what needs verification "
    "instead of inventing an announcement. " + NO_SEARCH_GUIDANCE
)
NEWS_IMPACT_PROMPT = (
    "You are a technology analyst. Analyze the supplied changes for developers and "
    "teams: who is affected, what changes, and migration considerations. "
    "Make uncertainty explicit."
)
NEWS_CTA_PROMPT = (
    "You are a tech editor. Given a news summary and impact analysis, write a brief, "
    "actionable next-steps section. Do not turn unverified claims into facts."
)
CONCEPT_ANALOGY_PROMPT = (
    "You are a tech educator. Explain the concept with a relatable analogy, "
    "then bridge to the technical idea."
)
CONCEPT_ARCHITECTURE_PROMPT = (
    "You are a software architect. Given the concept introduction, describe the "
    "technical architecture, component interactions, and common implementations."
)
CONCEPT_PROS_CONS_PROMPT = (
    "You are a pragmatic engineer. Given the concept and architecture, list honest "
    "pros and cons as two bullet lists."
)

RouterCallback = Callable[[str, dict[str, Any]], None]


class ContentRouter:
    """Classify a topic and run its specialized three-step chain."""

    def __init__(self, model: str, light_model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: RouterCallback = lambda _event, _data: None

    def _request(
        self, *, model: str, system: str, prompt: str, max_output_tokens: int = 8192, **kwargs: Any
    ) -> Any:
        logger.info("Calling %s", model)
        response = self.client.responses.create(
            model=model,
            max_output_tokens=max_output_tokens,
            instructions=system,
            input=prompt,
            reasoning={"effort": "none"},
            **kwargs,
        )
        if response.usage:
            self.token_tracker.track(response.usage)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise ValueError(
                f"DeepSeek response was {response.status} ({reason or 'unknown reason'})"
            )
        return response

    def _call_text(self, system: str, prompt: str, *, use_light: bool = False) -> str:
        model = self.light_model if use_light else self.model
        section_system = (
            f"{system}\n\nReturn only the requested section body. "
            "Do not add an article title or H1/H2 heading."
        )
        response = self._request(model=model, system=section_system, prompt=prompt)
        result = (response.output_text or "").strip()
        if not result:
            raise ValueError("DeepSeek produced empty text; aborting route.")
        return result

    def _classify(self, topic: str) -> dict[str, str]:
        response = self._request(
            model=self.light_model,
            system=CLASSIFY_SYSTEM_PROMPT,
            prompt=topic,
            max_output_tokens=512,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "function", "name": "classify_content"},
        )
        calls = [
            item
            for item in response.output
            if item.type == "function_call" and item.name == "classify_content"
        ]
        if len(calls) != 1:
            raise ValueError("Classifier did not return exactly one classify_content call.")
        try:
            classification = json.loads(calls[0].arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Classifier returned invalid JSON.") from exc
        if not isinstance(classification, dict):
            raise TypeError("Classifier returned a non-object result.")
        content_type = classification.get("content_type")
        reasoning = classification.get("reasoning")
        if content_type not in {"tutorial", "news", "concept"} or not isinstance(reasoning, str):
            raise ValueError("Classifier returned an invalid content type or reasoning.")
        return {"content_type": content_type, "reasoning": reasoning}

    def _step(self, name: str, system: str, prompt: str, *, use_light: bool = False) -> str:
        self._notify("step_start", {"name": name})
        result = self._call_text(system, prompt, use_light=use_light)
        self._notify("step_complete", {"name": name})
        return result

    def _chain_tutorial(self, topic: str) -> str:
        prerequisites = self._step(
            "Prerequisites", TUTORIAL_PREREQS_PROMPT, f"What prerequisites are needed for: {topic}"
        )
        steps = self._step(
            "Steps",
            TUTORIAL_STEPS_PROMPT,
            f"Topic: {topic}\n\nPrerequisites:\n{prerequisites}\n\nWrite the guide.",
            use_light=True,
        )
        troubleshooting = self._step(
            "Troubleshooting",
            TUTORIAL_TROUBLESHOOTING_PROMPT,
            f"Add troubleshooting for this tutorial:\n\n{steps}",
        )
        return (
            f"# {topic}\n\n## Prerequisites\n\n{prerequisites}\n\n"
            f"## Step-by-Step Guide\n\n{steps}\n\n"
            f"## Troubleshooting\n\n{troubleshooting}"
        )

    def _chain_news(self, topic: str) -> str:
        summary = self._step("Summary", NEWS_SUMMARY_PROMPT, f"Summarize the changes: {topic}")
        impact = self._step(
            "Impact",
            NEWS_IMPACT_PROMPT,
            f"Topic: {topic}\n\nAnalyze the impact of these changes:\n\n{summary}",
            use_light=True,
        )
        cta = self._step(
            "Call to Action",
            NEWS_CTA_PROMPT,
            f"News: {summary}\n\nImpact: {impact}\n\nWrite the call to action.",
        )
        return (
            f"# {topic}\n\n## What Changed\n\n{summary}\n\n"
            f"## Impact Analysis\n\n{impact}\n\n"
            f"## What You Should Do\n\n{cta}"
        )

    def _chain_concept(self, topic: str) -> str:
        analogy = self._step("Analogy", CONCEPT_ANALOGY_PROMPT, f"Explain: {topic}")
        architecture = self._step(
            "Architecture",
            CONCEPT_ARCHITECTURE_PROMPT,
            f"Concept intro: {analogy}\n\nDescribe the architecture of: {topic}",
            use_light=True,
        )
        pros_cons = self._step(
            "Pros/Cons",
            CONCEPT_PROS_CONS_PROMPT,
            f"Architecture: {architecture}\n\nList pros and cons of: {topic}",
        )
        return (
            f"# {topic}\n\n## Understanding the Concept\n\n{analogy}\n\n"
            f"## Architecture\n\n{architecture}\n\n"
            f"## Pros and Cons\n\n{pros_cons}"
        )

    def run(self, topic: str, on_event: RouterCallback | None = None) -> str:
        self._notify = on_event or (lambda _event, _data: None)
        self._notify("classify_start", {})
        classification = self._classify(topic)
        content_type = classification["content_type"]
        self.token_tracker.report()
        self._notify("classify_complete", classification)

        routes: dict[str, Callable[[str], str]] = {
            "tutorial": self._chain_tutorial,
            "news": self._chain_news,
            "concept": self._chain_concept,
        }
        self._notify("chain_start", {"content_type": content_type})
        result = routes[content_type](topic)
        self.token_tracker.report()
        self._notify("chain_complete", {"content_type": content_type})
        return result


def main() -> None:
    console = Console()
    token_tracker = OpenAITokenTracker()

    def on_router_event(event: str, data: dict[str, Any]) -> None:
        if event == "classify_start":
            console.print("\n[bold yellow]Step 1:[/bold yellow] Classifying topic...")
        elif event == "classify_complete":
            console.print(
                Panel(
                    f"[bold]{data['content_type'].upper()}[/bold]\n{data['reasoning']}",
                    title="Classification",
                    border_style="cyan",
                )
            )
        elif event == "chain_start":
            console.print(
                f"\n[bold yellow]Step 2:[/bold yellow] Running {data['content_type']} chain..."
            )
        elif event == "step_start":
            console.print(f"  [cyan]{data['name']}...[/cyan]")
        elif event == "step_complete":
            console.print("  [green]✓[/green] Done")

    header = Panel(
        "[bold cyan]Routing — DeepSeek Content Strategist[/bold cyan]\n\n"
        "Topic → Classifier → Specialized Chain → Article\n\n"
        "Tutorial: Prerequisites → Steps → Troubleshooting\n"
        "News: Changes → Impact → Call to Action\n"
        "Concept: Analogy → Architecture → Pros/Cons",
        title="Routing Demo",
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
                router = ContentRouter(MODEL, LIGHT_MODEL, token_tracker)
                result = router.run(topic, on_event=on_router_event)
                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = re.sub(r"[^\w]+", "_", topic.lower()).strip("_")[:50] or "article"
                path = OUTPUT_DIR / f"{slug}.md"
                path.write_text(result, encoding="utf-8")

                console.print("\n[bold blue]Final Article:[/bold blue]")
                console.print(Markdown(result))
                console.print(f"\n[dim]Saved to {path.resolve()}[/dim]")
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            except (OpenAIError, ValueError, TypeError, OSError, KeyError, EOFError) as exc:
                logger.error("Routing failed: %s", exc)
                console.print(f"\n[red]Error: {exc}[/red]")
            finally:
                token_tracker.reset()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
