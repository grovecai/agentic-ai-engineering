"""Choose the live model provider for tracing examples."""

import argparse
import os

import anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def provider_argument() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("anthropic", "deepseek"),
        default="anthropic",
        help="Provider for live calls (default: anthropic; no key uses sample/dry-run mode)",
    )
    return parser.parse_args().provider


def create_client(provider: str) -> tuple[anthropic.Anthropic | None, str]:
    """Return the selected client and model, or None for Anthropic offline mode."""
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
            os.environ.get("DEEPSEEK_TRACING_MODEL", "deepseek-flash"),
        )
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None, ANTHROPIC_MODEL
        return anthropic.Anthropic(), ANTHROPIC_MODEL
    raise ValueError(f"Unknown provider: {provider}")
