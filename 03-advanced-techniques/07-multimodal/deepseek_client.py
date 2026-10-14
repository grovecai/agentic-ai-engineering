"""Shared DeepSeek text client for the multimodal examples."""

import os

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())


def create_client() -> OpenAI:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or key == "your_deepseek_api_key_here":
        raise RuntimeError("Set DEEPSEEK_API_KEY in your environment or .env file")
    return OpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-flash")


def generate_text(client: OpenAI, instruction: str, user_text: str) -> str:
    response = client.chat.completions.create(
        model=model_name(),
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ],
        max_tokens=1024,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned no text")
    return content.strip()
