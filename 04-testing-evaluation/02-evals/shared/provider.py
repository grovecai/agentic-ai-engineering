"""Select the model provider used by the evaluation examples."""

import argparse
import os

import anthropic


def provider_argument() -> str:
    """Parse the provider flag shared by the three example scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("anthropic", "deepseek"),
        default="anthropic",
        help="Model provider for live evaluation (default: anthropic)",
    )
    return parser.parse_args().provider


def create_client(provider: str) -> tuple[anthropic.Anthropic | None, str]:
    """Return a client and model, or None for the original simulated mode."""
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for --provider deepseek")
        return (
            anthropic.Anthropic(
                api_key=api_key,
                base_url=os.environ.get(
                    "DEEPSEEK_ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"
                ),
            ),
            os.environ.get("DEEPSEEK_EVAL_MODEL", "deepseek-flash"),
        )

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None, "claude-sonnet-4-5-20250929"
        return anthropic.Anthropic(), "claude-sonnet-4-5-20250929"

    raise ValueError(f"Unknown provider: {provider}")
