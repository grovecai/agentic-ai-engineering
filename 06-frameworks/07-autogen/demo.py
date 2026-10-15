"""AutoGen AssistantAgent with a DeepSeek OpenAI-compatible model client."""

import asyncio
import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


async def main() -> None:
    client = OpenAIChatCompletionClient(
        model=MODEL,
        api_key=key(),
        base_url=BASE_URL,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )
    try:
        agent = AssistantAgent(
            "reference_agent",
            model_client=client,
            tools=[lookup_document],
            system_message=INSTRUCTIONS,
            max_tool_iterations=3,
        )
        result = await agent.run(task=QUESTION)
        print(result.messages[-1].content)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
