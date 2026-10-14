"""Write, evaluate, and refine a technical post with DeepSeek."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common import OpenAITokenTracker, interactive_menu, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

OUTPUT_DIR = Path("output")
MODEL = os.getenv("DEEPSEEK_EVALUATOR_MODEL", "deepseek-v4-pro")
LIGHT_MODEL = os.getenv("DEEPSEEK_EVALUATOR_LIGHT_MODEL", "deepseek-flash")
SCORE_THRESHOLD = 9.5
MAX_EVALUATIONS = 4

SUGGESTED_TOPICS = [
    "Building Event-Driven Microservices",
    "Edge Computing for Real-Time AI",
    "Modern CSS Layout Techniques",
    "Database Sharding Strategies",
]

NO_SEARCH_GUIDANCE = (
    "You have no web search access. Do not invent citations, benchmarks, release dates, "
    "or claim that time-sensitive facts have been verified. Mark uncertainty clearly."
)
WRITER_SYSTEM_PROMPT = (
    "You are a technical blog writer. Write a concise post with an introduction, "
    "3-5 useful sections, code examples where relevant, and a conclusion. "
    "Aim for under 1000 words. " + NO_SEARCH_GUIDANCE
)
EVALUATOR_SYSTEM_PROMPT = """\
You are a demanding technical editor assessing a technical article for engineers.
Judge only the supplied topic and draft. Read the whole draft before scoring.
Use the following observable checks; do not reward length, confident wording,
or the mere presence of headings or code. Do not compare the draft with earlier drafts.

Score each dimension independently with an integer from 1 to 10:

CLARITY
- Are terms defined before use, and are claims expressed without ambiguous referents?
- Can the reader follow each explanation and example without filling in missing steps?
- Do examples explain the mechanism rather than merely repeat the claim?

TECHNICAL ACCURACY
- Check for internal contradictions, incorrect causal claims, missing conditions,
  misleading absolutes, and code that appears syntactically or logically wrong.
- Distinguish facts stated in the draft from claims you can actually verify here.
  You have no web search, code execution, or external references. Never claim a
  time-sensitive fact, benchmark, version, or citation has been verified.
- If such a claim lacks support in the draft, flag it and suggest qualifying or
  sourcing it. A plausible claim is not proof of correctness.

STRUCTURE
- Does the introduction state a concrete problem and set up what the article covers?
- Do sections proceed in a useful order, with informative headings and transitions?
- Are repeated material and detours removed, and does the conclusion synthesize
  the practical takeaway rather than restate the introduction?

ENGAGEMENT
- Does the article use concrete decisions, examples, or trade-offs relevant to engineers?
- Is each section useful enough to keep reading, without generic hype or filler?
- Does the opening earn attention through a real problem rather than a broad slogan?

HUMAN VOICE
- Is the tone natural, direct, and consistent for a technical audience?
- Are sentence patterns varied without sacrificing precision?
- Flag templated transitions, exaggerated claims, empty adjectives, and robotic repetition.

Apply the same score anchors to every dimension:
10 = all checks pass, with unusually strong execution and no identifiable revision.
9 = all essential checks pass; only a tiny, localized polish remains.
7-8 = useful and generally sound, but one or more concrete minor weaknesses remain.
5-6 = substantial weakness that forces rereading, limits trust, or reduces usefulness.
3-4 = major errors, gaps, or disorganization; extensive revision is needed.
1-2 = largely unusable for the intended audience or topic.

Calibration rules:
- Start at 7 for a competent draft and move up or down only for observed evidence.
- Score 9 or 10 only when you can name what in the draft earns it. Do not give
  identical scores by default. A score of 10 requires no actionable improvement.
- An apparently broken code example caps technical_accuracy at 4. An unsupported
  precise or time-sensitive factual claim caps it at 6 until sourced or qualified.
- If the article misses a central part of the topic, cap structure and engagement
  at 6. Generic praise is not evidence for a high score.

For each dimension, put a brief, draft-specific justification in score_reasons.
List the most important issues in priority order. Each issue must identify the
section or quote a short phrase, explain the reader impact, and name the dimension.
Pair each issue with an actionable suggestion at the same array index: say exactly
what to add, remove, qualify, or rewrite. Avoid vague advice such as 'be clearer'.
If any score is below 10, provide at least one issue and suggestion. Do not invent
faults merely to fill the arrays. Always call evaluate_draft exactly once.
"""
REFINER_SYSTEM_PROMPT = (
    "You are a technical blog writer. Return the complete revised post. Address the "
    "editor's specific feedback while keeping a natural voice. " + NO_SEARCH_GUIDANCE
)

SCORE_KEYS = (
    "clarity",
    "technical_accuracy",
    "structure",
    "engagement",
    "human_voice",
)
SCORE_NAMES = (
    "Clarity",
    "Technical Accuracy",
    "Structure",
    "Engagement",
    "Human Voice",
)
EVALUATION_TOOL = {
    "type": "function",
    "name": "evaluate_draft",
    "description": "Score a technical blog post and provide concrete editing feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            **{
                key: {"type": "integer", "minimum": 1, "maximum": 10}
                for key in SCORE_KEYS
            },
            "score_reasons": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in SCORE_KEYS},
                "required": list(SCORE_KEYS),
                "additionalProperties": False,
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [*SCORE_KEYS, "score_reasons", "issues", "suggestions"],
        "additionalProperties": False,
    },
}

EvaluatorCallback = Callable[[str, dict[str, Any]], None]


def _validate_evaluation(value: Any) -> dict[str, Any]:
    """Check model output before using it to control the loop."""
    if not isinstance(value, dict):
        raise ValueError("Evaluator returned a non-object result.")
    for key in SCORE_KEYS:
        score = value.get(key)
        if type(score) is not int or not 1 <= score <= 10:
            raise ValueError(f"Evaluator returned an invalid {key} score.")
    reasons = value.get("score_reasons")
    if not isinstance(reasons, dict) or any(
        not isinstance(reasons.get(key), str) or not reasons[key].strip() for key in SCORE_KEYS
    ):
        raise ValueError("Evaluator returned invalid score reasons.")
    for key in ("issues", "suggestions"):
        items = value.get(key)
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError(f"Evaluator returned invalid {key}.")
    if len(value["issues"]) != len(value["suggestions"]):
        raise ValueError("Evaluator returned unpaired issues and suggestions.")
    if any(value[key] < 10 for key in SCORE_KEYS) and not value["suggestions"]:
        raise ValueError("Evaluator returned no revision suggestions for a sub-10 score.")
    return value


class EvaluatorOptimizer:
    """Use separate writing, evaluation, and revision calls."""

    def __init__(self, model: str, light_model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.light_model = light_model
        self.token_tracker = token_tracker
        self._notify: EvaluatorCallback = lambda _event, _data: None

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
            self.token_tracker.track(response.usage)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise ValueError(
                f"DeepSeek response was {response.status} ({reason or 'unknown reason'})"
            )
        return response

    def _call_text(self, *, model: str, system: str, prompt: str) -> str:
        response = self._request(
            model=model, system=system, prompt=prompt, max_output_tokens=8192
        )
        result = (response.output_text or "").strip()
        if not result:
            raise ValueError("DeepSeek produced empty text.")
        return result

    def _evaluate(self, topic: str, draft: str) -> dict[str, Any]:
        prompt = f"Topic: {topic}\n\nDraft to evaluate:\n\n{draft}"
        for attempt in range(2):
            response = self._request(
                model=self.light_model,
                system=EVALUATOR_SYSTEM_PROMPT,
                prompt=prompt,
                max_output_tokens=3072,
                tools=[EVALUATION_TOOL],
                tool_choice={"type": "function", "name": "evaluate_draft"},
            )
            try:
                calls = [
                    item
                    for item in response.output
                    if item.type == "function_call" and item.name == "evaluate_draft"
                ]
                if len(calls) != 1:
                    raise ValueError("Evaluator did not return exactly one evaluation.")
                evaluation = json.loads(calls[0].arguments)
                return _validate_evaluation(evaluation)
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt:
                    raise ValueError("Evaluator failed to return a valid evaluation.") from exc
                logger.warning("Invalid evaluator output; retrying once: %s", exc)
                prompt += (
                    "\n\nReturn exactly one evaluate_draft call. All scores must be integers "
                    "from 1 to 10. Supply nonempty score_reasons for every dimension. "
                    "issues and suggestions must be equally sized arrays of strings."
                )
        raise AssertionError("Unreachable evaluator state")

    def run(self, topic: str, on_event: EvaluatorCallback | None = None) -> str:
        self._notify = on_event or (lambda _event, _data: None)
        self._notify("write_start", {})
        draft = self._call_text(model=self.light_model, system=WRITER_SYSTEM_PROMPT, prompt=topic)
        self._notify("draft_complete", {"chars": len(draft)})

        # Evaluate the initial draft and each revision, at most four times total.
        for iteration in range(MAX_EVALUATIONS):
            self._notify("evaluate_start", {"iteration": iteration + 1})
            evaluation = self._evaluate(topic, draft)
            scores = dict(zip(SCORE_NAMES, (evaluation[key] for key in SCORE_KEYS)))
            average = sum(scores.values()) / len(scores)
            self.token_tracker.report()
            if evaluation["suggestions"]:
                for number, suggestion in enumerate(evaluation["suggestions"], start=1):
                    logger.info(
                        "Evaluation %d suggestion %d: %s", iteration + 1, number, suggestion
                    )
            else:
                logger.info("Evaluation %d returned no revision suggestions", iteration + 1)
            self._notify(
                "evaluation_complete",
                {
                    "scores": scores,
                    "score_reasons": evaluation["score_reasons"],
                    "avg": average,
                    "issues": evaluation["issues"],
                    "suggestions": evaluation["suggestions"],
                },
            )
            if average >= SCORE_THRESHOLD:
                self._notify("threshold_met", {"avg": average})
                break
            if iteration == MAX_EVALUATIONS - 1:
                self._notify("max_evaluations", {"avg": average})
                break

            self._notify("refining", {"avg": average})
            prompt = (
                f"Topic: {topic}\n\nPrevious draft:\n{draft}\n\n"
                f"Issues: {json.dumps(evaluation['issues'], ensure_ascii=False)}\n"
                f"Suggestions: {json.dumps(evaluation['suggestions'], ensure_ascii=False)}\n\n"
                "Revise the complete draft to address the feedback."
            )
            draft = self._call_text(model=self.model, system=REFINER_SYSTEM_PROMPT, prompt=prompt)
            self._notify("draft_complete", {"chars": len(draft)})
        return draft


def main() -> None:
    console = Console()
    token_tracker = OpenAITokenTracker()

    def on_event(event: str, data: dict[str, Any]) -> None:
        if event == "write_start":
            console.print("\n[bold yellow]Writing:[/bold yellow] Generating initial draft...")
        elif event == "draft_complete":
            console.print(f"  [green]✓[/green] Draft: {data['chars']} chars")
        elif event == "evaluate_start":
            console.print(
                f"\n[bold yellow]Evaluating:[/bold yellow] Round {data['iteration']}"
                f"/{MAX_EVALUATIONS}..."
            )
        elif event == "evaluation_complete":
            table = Table(title=f"Evaluation (avg: {data['avg']:.1f}/10)")
            table.add_column("Dimension", style="cyan")
            table.add_column("Score", justify="center")
            table.add_column("Evidence")
            for name, key in zip(SCORE_NAMES, SCORE_KEYS):
                table.add_row(name, f"{data['scores'][name]}/10", data["score_reasons"][key])
            console.print(table)
            for key, label in (("issues", "Issues"), ("suggestions", "Suggestions")):
                if data[key]:
                    console.print(f"[bold yellow]{label}:[/bold yellow]")
                    for item in data[key]:
                        console.print(f"  • {item}")
        elif event == "threshold_met":
            console.print(f"[green]Score {data['avg']:.1f} >= {SCORE_THRESHOLD} — done![/green]")
        elif event == "refining":
            console.print("[yellow]Below threshold — revising...[/yellow]")
        elif event == "max_evaluations":
            console.print(f"[yellow]Max evaluations reached (score: {data['avg']:.1f})[/yellow]")

    header = Panel(
        "[bold cyan]Evaluator-Optimizer — DeepSeek[/bold cyan]\n\n"
        "Topic → Writer → Evaluator → Refiner → Evaluator\n"
        f"Stop at average score >= {SCORE_THRESHOLD} or after {MAX_EVALUATIONS} evaluations.\n"
        "No web search; current facts are not verified.",
        title="Evaluator-Optimizer",
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
                result = EvaluatorOptimizer(MODEL, LIGHT_MODEL, token_tracker).run(
                    topic, on_event=on_event
                )
                OUTPUT_DIR.mkdir(exist_ok=True)
                slug = topic.lower().replace(" ", "_")[:50]
                path = OUTPUT_DIR / f"{slug}.md"
                path.write_text(result, encoding="utf-8")
                console.print("\n[bold blue]Final Article:[/bold blue]")
                console.print(Markdown(result))
                console.print(f"\n[dim]Saved to {path.resolve()}[/dim]")
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            except Exception as exc:
                logger.error("Evaluator-optimizer failed: %s", exc)
                console.print(f"\n[red]Error: {exc}[/red]")
            finally:
                token_tracker.reset()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
