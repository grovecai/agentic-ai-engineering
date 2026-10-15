"""Semantic Kernel chat agent with a local lookup plugin."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


class Documents:
    @kernel_function(description="Read a local operations document by ID")
    def lookup_document(self, doc_id: Annotated[str, "runbook or incident"]) -> str:
        return lookup_document(doc_id)


async def main() -> None:
    client = AsyncOpenAI(api_key=key(), base_url=BASE_URL)
    try:
        service = OpenAIChatCompletion(ai_model_id=MODEL, async_client=client)
        agent = ChatCompletionAgent(
            service=service,
            name="reference_agent",
            instructions=INSTRUCTIONS,
            plugins=[Documents()],
        )
        async for response in agent.invoke(messages=QUESTION):
            print(response.content)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
