"""Small DeepSeek tool loop shared by the loop-engineering lessons."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DOCUMENTS = {
    "runbook": "Deploy changes with a health check, then watch error rate and latency. Roll back if either breaches its threshold.",
    "incident": "For an incident, record a trace ID, assign an owner, mitigate impact, and write a short follow-up after recovery.",
}


def client_from_env() -> OpenAI:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY to run this live example")
    return OpenAI(api_key=key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def lookup(args: dict[str, Any]) -> str:
    doc_id = args.get("doc_id")
    if not isinstance(doc_id, str) or doc_id not in DOCUMENTS:
        return "Unknown document ID. Available: runbook, incident."
    return f"{doc_id}: {DOCUMENTS[doc_id]}"


LOOKUP = Tool(
    "lookup",
    "Read a local operations document by ID. Cite its ID in your answer.",
    {
        "type": "object",
        "properties": {"doc_id": {"type": "string", "enum": list(DOCUMENTS)}},
        "required": ["doc_id"],
    },
    lookup,
)


@dataclass
class Hooks:
    before: list[Callable[[str, dict[str, Any]], None]] = field(default_factory=list)
    after: list[Callable[[str, dict[str, Any], Any], None]] = field(default_factory=list)
    stop: list[Callable[[str], None]] = field(default_factory=list)


class Agent:
    def __init__(
        self, client: Any, tools: list[Tool], hooks: Hooks | None = None, max_steps: int = 5
    ):
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        self.hooks = hooks or Hooks()
        self.max_steps = max_steps
        self.last_usage = {"input": 0, "output": 0}
        self.last_messages: list[Any] = []

    def run(
        self,
        question: str,
        system: str = "Use tools for local facts and cite document IDs.",
        history: list[Any] | None = None,
    ) -> str:
        messages: list[Any] = list(history) if history else [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": question})
        try:
            for _ in range(self.max_steps):
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=[tool.schema() for tool in self.tools.values()] or None,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                usage = response.usage
                if usage:
                    self.last_usage["input"] += usage.prompt_tokens or 0
                    self.last_usage["output"] += usage.completion_tokens or 0
                message = response.choices[0].message
                calls = message.tool_calls or []
                if not calls:
                    answer = message.content or ""
                    if not answer:
                        raise RuntimeError("Model returned no answer")
                    messages.append(message.model_dump(exclude_none=True))
                    return answer
                messages.append(message.model_dump(exclude_none=True))
                for call in calls:
                    name = call.function.name
                    if name not in self.tools:
                        result: Any = {"error": f"Unknown tool: {name}"}
                    else:
                        try:
                            args = json.loads(call.function.arguments)
                            if not isinstance(args, dict):
                                raise TypeError("Tool arguments must be an object")
                            for hook in self.hooks.before:
                                hook(name, args)
                            result = self.tools[name].execute(args)
                            for hook in self.hooks.after:
                                hook(name, args, result)
                        except (ValueError, TypeError, PermissionError) as exc:
                            result = {"error": str(exc)}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
            raise RuntimeError(f"Agent exceeded {self.max_steps} model steps")
        finally:
            self.last_messages = messages
            for hook in self.hooks.stop:
                hook(question)
