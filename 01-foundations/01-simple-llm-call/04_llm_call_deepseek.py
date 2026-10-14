"""
Simple LLM Call (DeepSeek)

Demonstrates a basic call to DeepSeek's OpenAI-compatible Responses API.
Shows separation of agent logic from orchestration.
"""

import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from common.logging_config import setup_logging

# Load environment variables from root .env file
load_dotenv(find_dotenv())

# Configure logging
logger = setup_logging(__name__)


class LLMClient:
    """
    Simple agent that makes basic LLM calls to DeepSeek.

    Encapsulates all agent logic including API interaction.
    """

    def __init__(self, model: str):
        """Initialize the agent using the OpenAI SDK with DeepSeek's endpoint."""
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model
        self.system_prompt = "You are a helpful AI assistant. Provide clear and concise answers."

    def run(self, prompt: str) -> str:
        """Execute the agent with a given prompt."""
        logger.info(f"Calling model: {self.model}")

        response = self.client.responses.create(
            model=self.model,
            temperature=0.1,
            max_output_tokens=1024,
            instructions=self.system_prompt,
            input=prompt,
            # Disable thinking for this basic call so temperature takes effect.
            reasoning={"effort": "none"},
        )

        # Log token usage if available
        if response.usage:
            logger.info(
                f"Token usage - Input: {response.usage.input_tokens}, "
                f"Output: {response.usage.output_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

        # Extract and return response
        return response.output_text or ""


def main() -> None:
    """Set up the agent and coordinate execution flow."""
    agent = LLMClient(os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))

    prompt = "Explain what an AI agent is in 2-3 sentences."
    logger.info(f"👤 User: {prompt}")

    # Call LLM
    response = agent.run(prompt)

    # Display results
    logger.info(f"🤖 Response: {response}")


if __name__ == "__main__":
    main()
