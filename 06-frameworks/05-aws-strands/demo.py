"""Strands model-driven agent with an OpenAI-compatible DeepSeek model."""

import sys
from pathlib import Path

from strands import Agent, tool
from strands.models.openai import OpenAIModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


@tool
def lookup(doc_id: str) -> str:
    """Read the local runbook or incident document by ID."""
    return lookup_document(doc_id)


if __name__ == "__main__":
    model = OpenAIModel(
        client_args={"api_key": key(), "base_url": BASE_URL},
        model_id=MODEL,
        params={"extra_body": {"thinking": {"type": "disabled"}}},
    )
    agent = Agent(model=model, tools=[lookup], system_prompt=INSTRUCTIONS, callback_handler=None)
    print(agent(QUESTION))
