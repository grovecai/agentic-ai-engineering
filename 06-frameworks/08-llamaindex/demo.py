"""LlamaIndex FunctionAgent with an OpenAI-compatible DeepSeek LLM."""

import asyncio
import sys
from pathlib import Path

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


async def main() -> None:
    llm = OpenAILike(
        model=MODEL,
        api_base=BASE_URL,
        api_key=key(),
        is_chat_model=True,
        is_function_calling_model=True,
    )
    agent = FunctionAgent(
        tools=[FunctionTool.from_defaults(fn=lookup_document)], llm=llm, system_prompt=INSTRUCTIONS
    )
    result = await agent.run(user_msg=QUESTION)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
