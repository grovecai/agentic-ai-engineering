"""Pydantic AI: typed document ID, tool registration and result validation."""

import sys
from pathlib import Path
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document

model = OpenAIChatModel(MODEL, provider=OpenAIProvider(base_url=BASE_URL, api_key=key()))
agent = Agent(model, instructions=INSTRUCTIONS)


@agent.tool_plain
def lookup(doc_id: Literal["runbook", "incident"]) -> str:
    """Read one local document by ID."""
    return lookup_document(doc_id)


if __name__ == "__main__":
    print(agent.run_sync(QUESTION).output)
