"""Structured output with DeepSeek: tool calls, JSON Schema, retries, and batches."""

import json
import os
from typing import Any, Literal

from common import OpenAITokenTracker, interactive_menu, setup_logging
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

load_dotenv(find_dotenv())
logger = setup_logging(__name__)

MODEL = os.getenv("DEEPSEEK_STRUCTURED_MODEL", "deepseek-flash")
SYSTEM_PROMPT = (
    "Analyze customer support tickets accurately. Return only facts supported by the ticket. "
    "If escalation is required, give a specific escalation reason."
)


class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "account", "feature_request", "general"]
    priority: Literal["critical", "high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative", "frustrated"]
    summary: str = Field(description="One-sentence summary of the ticket")


class Entity(BaseModel):
    name: str
    type: Literal["product", "feature", "error_code", "account_id", "person"]
    context: str


class ActionItem(BaseModel):
    action: str
    assignee: Literal["support", "engineering", "billing", "account_manager"]
    urgency: Literal["immediate", "next_business_day", "backlog"]


class TicketAnalysis(BaseModel):
    classification: TicketClassification
    entities: list[Entity]
    action_items: list[ActionItem]
    requires_escalation: bool
    escalation_reason: str | None = None
    customer_tier: Literal["free", "pro", "enterprise"] | None = None

    @model_validator(mode="after")
    def check_escalation(self) -> "TicketAnalysis":
        if self.requires_escalation and not self.escalation_reason:
            raise ValueError("escalation_reason is required when requires_escalation is true")
        return self


class PriorityDistribution(BaseModel):
    critical: int
    high: int
    medium: int
    low: int


class TicketBatch(BaseModel):
    analyses: list[TicketAnalysis]
    batch_summary: str
    priority_distribution: PriorityDistribution


SAMPLE_TICKETS = [
    (
        "Subject: Double charged for Pro subscription\n"
        "I was charged twice for Pro this month: $49.99 on Jan 3 and Jan 5. "
        "My account ID is ACC-78234. This has happened three times. "
        "Please refund the duplicate charge today or I will cancel."
    ),
    (
        "Subject: SSO blocker during enterprise evaluation\n"
        "Our team of 200 engineers is evaluating your product. Okta SSO works, "
        "but SCIM group sync returns 500 for groups over 50 members (SCIM-ERR-4012). "
        "Can we discuss volume pricing? Contact Sarah Chen, VP Engineering."
    ),
]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Make Pydantic's schema suitable for constrained JSON output.

    Optional fields remain nullable, but must be present. DeepSeek's schema output
    requires closed objects; Pydantic still checks business rules afterward.
    """
    schema = model.model_json_schema()

    def close_objects(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close_objects(value)
        elif isinstance(node, list):
            for value in node:
                close_objects(value)

    close_objects(schema)
    return schema


class StructuredExtractor:
    def __init__(self, model: str, token_tracker: OpenAITokenTracker):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.token_tracker = token_tracker

    def _request(self, prompt: str, **kwargs: Any) -> Any:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            reasoning={"effort": "none"},
            max_output_tokens=4096,
            **kwargs,
        )
        if response.usage:
            self.token_tracker.track(response.usage)
        if response.status != "completed":
            reason = getattr(response.incomplete_details, "reason", None)
            raise ValueError(f"DeepSeek response was {response.status}: {reason}")
        return response

    def extract_with_tool(self, text: str) -> TicketClassification:
        """Force one function call and validate its arguments with Pydantic."""
        name = "classify_ticket"
        response = self._request(
            f"Classify this support ticket:\n\n{text}",
            tools=[
                {
                    "type": "function",
                    "name": name,
                    "description": "Return the support ticket classification.",
                    "parameters": _strict_schema(TicketClassification),
                }
            ],
            tool_choice={"type": "function", "name": name},
        )
        calls = [
            item for item in response.output if item.type == "function_call" and item.name == name
        ]
        if len(calls) != 1:
            raise ValueError(f"Expected one {name} call, got {len(calls)}")
        return TicketClassification.model_validate_json(calls[0].arguments)

    def extract_with_schema(
        self, text: str, model: type[BaseModel] = TicketClassification
    ) -> BaseModel:
        """Use DeepSeek Responses API JSON Schema output, then validate locally."""
        response = self._request(
            f"Analyze this support ticket:\n\n{text}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": model.__name__.lower(),
                    "schema": _strict_schema(model),
                }
            },
        )
        if not response.output_text:
            raise ValueError("DeepSeek returned empty structured output")
        return model.model_validate_json(response.output_text)

    def extract_with_retry(self, text: str, max_attempts: int = 3) -> TicketAnalysis:
        """Feed Pydantic validation errors and the prior JSON back for repair."""
        prompt = f"Analyze this support ticket:\n\n{text}"
        for attempt in range(1, max_attempts + 1):
            response = self._request(
                prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ticket_analysis",
                        "schema": _strict_schema(TicketAnalysis),
                    }
                },
            )
            raw = response.output_text
            if not raw:
                raise ValueError("DeepSeek returned empty structured output")
            try:
                return TicketAnalysis.model_validate_json(raw)
            except ValidationError as exc:
                logger.warning("Attempt %d failed Pydantic validation: %s", attempt, exc)
                if attempt == max_attempts:
                    raise
                prompt = (
                    f"Original ticket:\n{text}\n\nPrevious JSON:\n{raw}\n\n"
                    f"Validation errors:\n{exc}\n\nCorrect the JSON. If escalation is true, "
                    "give a non-empty escalation_reason."
                )
        raise RuntimeError("Unreachable retry state")

    def extract_batch(self, texts: list[str]) -> TicketBatch:
        numbered = "\n\n".join(f"Ticket {i}:\n{text}" for i, text in enumerate(texts, 1))
        response = self._request(
            f"Analyze all {len(texts)} tickets. Include exactly one analysis per ticket "
            f"and count priorities across the batch.\n\n{numbered}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ticket_batch",
                    "schema": _strict_schema(TicketBatch),
                }
            },
        )
        if not response.output_text:
            raise ValueError("DeepSeek returned empty batch output")
        result = TicketBatch.model_validate_json(response.output_text)
        if len(result.analyses) != len(texts):
            raise ValueError(f"Expected {len(texts)} analyses, got {len(result.analyses)}")
        counts = result.priority_distribution
        if sum((counts.critical, counts.high, counts.medium, counts.low)) != len(texts):
            raise ValueError("Priority distribution does not match batch size")
        return result


def _display(console: Console, title: str, result: BaseModel) -> None:
    formatted = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    console.print(Panel(Syntax(formatted, "json", theme="monokai"), title=title))


def main() -> None:
    console = Console()
    tracker = OpenAITokenTracker()
    extractor = StructuredExtractor(MODEL, tracker)
    choices = [
        "1: Function call as structured output",
        "2: Native JSON Schema output",
        "3: Pydantic validation and retry",
        "4: Batch extraction",
    ]
    header = Panel(
        "[bold cyan]Structured Output — DeepSeek[/bold cyan]\n"
        "Tool call → JSON Schema → validation retry → batch extraction",
        title="Advanced Techniques",
    )
    try:
        while True:
            choice = interactive_menu(console, choices, title="Select a Technique", header=header)
            if not choice:
                break
            try:
                if choice == choices[0]:
                    result = extractor.extract_with_tool(SAMPLE_TICKETS[0])
                elif choice == choices[1]:
                    result = extractor.extract_with_schema(SAMPLE_TICKETS[0])
                elif choice == choices[2]:
                    result = extractor.extract_with_retry(SAMPLE_TICKETS[1])
                else:
                    result = extractor.extract_batch(SAMPLE_TICKETS)
                _display(console, choice, result)
            except (OpenAIError, ValidationError, ValueError, KeyError) as exc:
                logger.error("Extraction failed: %s", exc)
                console.print(Panel(str(exc), title="Extraction failed", border_style="red"))
            finally:
                tracker.report()
                tracker.reset()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
