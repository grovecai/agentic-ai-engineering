"""Shared, illustrative DeepSeek prices and usage accounting.

Prices are USD per million tokens. Update from the official pricing page when
running these demos; the selected tier must match the time of the API calls.
"""

import os
from dataclasses import dataclass
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

FLASH = "deepseek-flash"
PRO = "deepseek-v4-pro"
PRICE_TIER = os.getenv("DEEPSEEK_PRICE_TIER", "off_peak").lower()
if PRICE_TIER not in {"off_peak", "peak"}:
    raise ValueError("DEEPSEEK_PRICE_TIER must be 'off_peak' or 'peak'")

# https://api-docs.deepseek.com/quick_start/pricing/ (October 2026)
OFF_PEAK_PRICES = {
    FLASH: (0.003, 0.15, 0.60),  # cached input, uncached input, output
    PRO: (0.022, 0.66, 1.98),
}


def client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int

    @classmethod
    def from_response(cls, response: Any) -> "Usage":
        usage = response.usage
        if usage is None:
            raise ValueError("DeepSeek response did not include token usage")
        # Chat Completions uses prompt/completion_tokens; Responses uses
        # input/output_tokens. Both APIs report cached input in their details.
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            details = getattr(usage, "input_tokens_details", None)
        if input_tokens is None or output_tokens is None:
            raise ValueError("DeepSeek response did not include input/output token counts")
        cached = getattr(details, "cached_tokens", None)
        if cached is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        if not 0 <= cached <= input_tokens:
            raise ValueError("Invalid cached token count in DeepSeek response")
        return cls(input_tokens, cached, output_tokens)

    @property
    def uncached_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens


def cost(model: str, usage: Usage, *, ignore_cache: bool = False) -> float:
    cached_price, uncached_price, output_price = OFF_PEAK_PRICES[model]
    multiplier = 2 if PRICE_TIER == "peak" else 1
    cached = 0 if ignore_cache else usage.cached_tokens
    uncached = usage.input_tokens - cached
    return multiplier * (
        cached * cached_price + uncached * uncached_price + usage.output_tokens * output_price
    ) / 1_000_000
