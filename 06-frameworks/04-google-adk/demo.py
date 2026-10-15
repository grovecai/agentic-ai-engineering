"""Google ADK runner with a LiteLLM bridge to DeepSeek."""

import asyncio
import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


async def main() -> None:
    agent = Agent(
        name="reference_agent",
        model=LiteLlm(model=f"openai/{MODEL}", api_base=BASE_URL, api_key=key()),
        instruction=INSTRUCTIONS,
        tools=[lookup_document],
    )
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="framework_comparison", user_id="demo", session_id="one")
    runner = Runner(agent=agent, app_name="framework_comparison", session_service=sessions)
    async for event in runner.run_async(
        user_id="demo",
        session_id="one",
        new_message=types.Content(role="user", parts=[types.Part(text=QUESTION)]),
    ):
        if event.is_final_response() and event.content:
            print("".join(part.text or "" for part in event.content.parts))


if __name__ == "__main__":
    asyncio.run(main())
