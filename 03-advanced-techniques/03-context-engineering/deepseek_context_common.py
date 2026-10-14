"""Shared budget display and conservative local token estimates for DeepSeek demos."""

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def estimate_tokens(value: Any) -> int:
    """Approximate tokens from UTF-8 bytes; this is not the DeepSeek tokenizer."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


def estimate_input(instructions: str, messages: list[dict], tools: list[dict] | None = None) -> int:
    """Estimate total request input, including fixed instructions and tool schemas."""
    return (
        estimate_tokens(instructions)
        + estimate_tokens(tools or [])
        + sum(estimate_tokens(message) + 8 for message in messages)
    )


@dataclass
class ContextBudget:
    max_context: int
    fixed_estimate: int
    response_reserve: int

    @property
    def history_budget(self) -> int:
        return self.max_context - self.fixed_estimate - self.response_reserve


@dataclass
class TokenSnapshot:
    fixed_estimate: int
    history_estimate: int
    history_budget: int
    response_reserve: int
    message_count: int
    compression_count: int
    last_actual_input: int | None


def render_budget(console: Console, snapshot: TokenSnapshot) -> None:
    ratio = snapshot.history_estimate / max(1, snapshot.history_budget)
    color = "green" if ratio < 0.7 else "yellow" if ratio < 0.9 else "red"
    filled = min(25, int(ratio * 25))
    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (25 - filled)}[/dim]"
    table = Table(show_header=False, box=None)
    table.add_column("Component")
    table.add_column("Tokens", justify="right")
    table.add_column("Usage", min_width=25)
    table.add_row("System + tools estimate", str(snapshot.fixed_estimate), "")
    table.add_row(
        "History estimate",
        f"{snapshot.history_estimate} / {snapshot.history_budget}",
        bar,
    )
    table.add_row("Response reserve", str(snapshot.response_reserve), "")
    if snapshot.last_actual_input is not None:
        table.add_row("Last actual API input", str(snapshot.last_actual_input), "")
    console.print(
        Panel(
            table,
            title="Context Budget (preflight numbers are estimates)",
            subtitle=(
                f"Messages: {snapshot.message_count} | Compressions: {snapshot.compression_count}"
            ),
        )
    )
