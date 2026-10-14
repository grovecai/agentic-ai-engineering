"""Draft an email with DeepSeek and pause for human approval or revision."""

import os
from collections.abc import Callable

from common import OpenAITokenTracker, interactive_menu, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = os.getenv("DEEPSEEK_HUMAN_LOOP_MODEL", "deepseek-flash")
MAX_REVISIONS = 2

SUGGESTED_SCENARIOS = [
    "Decline a job offer politely — grateful but chose another opportunity",
    "Ask your team to work overtime this weekend — critical deadline, apologetic tone",
    "Request a meeting with a VP to discuss budget — formal, data-driven",
    "Follow up on an unanswered proposal — persistent but respectful",
]

SYSTEM_PROMPT = (
    "You are a professional email writer. Write clear, concise emails that match "
    "the requested tone. Output only the email — subject line, then body. "
    "No meta-commentary. Keep it under 300 words. Do not invent names or facts."
)
REVISE_SYSTEM_PROMPT = (
    "You are a professional email writer. Revise the latest email draft using the "
    "original scenario and all human feedback in order. Preserve earlier requested "
    "changes unless newer feedback explicitly conflicts; newer feedback wins. "
    "Preserve details the human did not ask you to change. Return only the complete "
    "revised email — subject line, then body. No explanation."
)

# (title, content, question) -> (approved, feedback or replacement text)
CheckpointFn = Callable[[str, str, str], tuple[bool, str]]


class EmailDrafter:
    """Draft and revise emails with injectable human checkpoints."""

    def __init__(self, model: str, token_tracker: OpenAITokenTracker) -> None:
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.token_tracker = token_tracker

    def _call_llm(self, system: str, prompt: str) -> str:
        logger.info("Calling %s", self.model)
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=1024,
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
            raise ValueError("DeepSeek produced an empty email.")
        return result

    def _draft(self, scenario: str) -> str:
        return self._call_llm(SYSTEM_PROMPT, f"Write an email for this scenario: {scenario}")

    def _revise(self, scenario: str, draft: str, feedback_history: list[str]) -> str:
        feedback = "\n".join(
            f"{number}. {item}" for number, item in enumerate(feedback_history, start=1)
        )
        prompt = (
            f"Original scenario:\n{scenario}\n\n"
            f"Latest email draft:\n{draft}\n\n"
            f"Human feedback, oldest to newest:\n{feedback}"
        )
        return self._call_llm(REVISE_SYSTEM_PROMPT, prompt)

    def run(self, scenario: str, *, checkpoint_fn: CheckpointFn | None = None) -> str:
        """Return the approved, human-edited, or last revised email; never send it."""
        check = checkpoint_fn or (lambda _title, _content, _question: (True, ""))

        logger.info("Generating draft for: %s", scenario)
        draft = self._draft(scenario)
        self.token_tracker.report()
        approved, feedback = check(
            "Draft Review",
            draft,
            "Does this email look right? Approve, edit, or reject with feedback.",
        )
        if approved:
            return feedback or draft

        feedback_history = [feedback]
        for revision in range(1, MAX_REVISIONS + 1):
            logger.info("Revising draft (round %d/%d)", revision, MAX_REVISIONS)
            draft = self._revise(scenario, draft, feedback_history)
            self.token_tracker.report()
            approved, feedback = check(
                f"Revision Review ({revision}/{MAX_REVISIONS})",
                draft,
                "Better? Approve, edit, or reject with more feedback.",
            )
            if approved:
                return feedback or draft
            feedback_history.append(feedback)

        logger.info("Max revisions reached; returning the last draft for further human review")
        return draft


def human_checkpoint(console: Console, title: str, content: str, question: str) -> tuple[bool, str]:
    """Ask for approval, feedback, or a complete replacement in the terminal."""
    console.print(Panel(content, title=f"Checkpoint: {title}", border_style="bright_magenta"))
    console.print(f"\n[bold magenta]{question}[/bold magenta]")
    console.print("[dim](y)es / (n)o with feedback / (e)dit to provide replacement[/dim]")

    while True:
        console.print("[bold magenta]> [/bold magenta]", end="")
        response = input().strip().lower()
        if response in ("y", "yes", ""):
            return True, ""
        if response in ("e", "edit"):
            console.print("[dim]Enter replacement (blank line to finish):[/dim]")
            lines: list[str] = []
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            replacement = "\n".join(lines).strip()
            if replacement:
                return True, replacement
            console.print("[yellow]Replacement cannot be empty.[/yellow]")
        elif response in ("n", "no"):
            console.print("[dim]Enter feedback:[/dim] ", end="")
            feedback = input().strip()
            if feedback:
                return False, feedback
            console.print("[yellow]Feedback cannot be empty.[/yellow]")
        else:
            console.print("[yellow]Choose y, n, or e.[/yellow]")


def main() -> None:
    console = Console()
    token_tracker = OpenAITokenTracker()

    def checkpoint_fn(title: str, content: str, question: str) -> tuple[bool, str]:
        return human_checkpoint(console, title, content, question)

    header = Panel(
        "[bold cyan]Human-in-the-Loop — DeepSeek[/bold cyan]\n\n"
        "DeepSeek drafts an email → you review at checkpoints:\n"
        "1. After draft — right tone and content?\n"
        "2. After revision — feedback addressed?\n\n"
        "Options: (y)es approve, (n)o + feedback, (e)dit replacement\n"
        f"Max {MAX_REVISIONS} revisions per email.",
        title="Human-in-the-Loop",
    )

    try:
        while True:
            scenario = interactive_menu(
                console,
                SUGGESTED_SCENARIOS,
                title="Select an Email Scenario",
                header=header,
                allow_custom=True,
                custom_prompt="Describe your email scenario",
            )
            if not scenario:
                break

            console.print(f"\n[bold green]Scenario:[/bold green] {scenario}")
            try:
                result = EmailDrafter(MODEL, token_tracker).run(
                    scenario, checkpoint_fn=checkpoint_fn
                )
                console.print("\n[bold blue]Final Email:[/bold blue]")
                console.print(Panel(result, border_style="green"))
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
            except Exception as exc:
                logger.error("Email drafting failed: %s", exc)
                console.print(f"\n[red]Error: {exc}[/red]")
            finally:
                token_tracker.reset()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
