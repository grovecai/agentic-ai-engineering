"""Compare first and cached answer; enforce an output-token cap."""

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Settings, SupportService


def route_model(question: str) -> str:
    """A transparent teaching rule; replace with eval-based routing in real traffic."""
    return (
        os.getenv("DEEPSEEK_COMPLEX_MODEL", "deepseek-v4-pro")
        if len(question) > 120
        else os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    )


if __name__ == "__main__":
    question = "How long does shipping take?"
    selected_model = route_model(question)
    service = SupportService(replace(Settings.from_env(), model=selected_model))
    print("Selected model:", selected_model)
    for _ in range(2):
        print(service.answer(question, use_cache=True))
    print(
        "Actual model token usage:",
        service.metrics["input_tokens"],
        service.metrics["output_tokens"],
    )
    input_rate = os.getenv("DEEPSEEK_INPUT_USD_PER_MILLION")
    output_rate = os.getenv("DEEPSEEK_OUTPUT_USD_PER_MILLION")
    if input_rate and output_rate:
        estimate = (
            service.metrics["input_tokens"] * float(input_rate)
            + service.metrics["output_tokens"] * float(output_rate)
        ) / 1_000_000
        print(f"Estimated cost of this model request: ${estimate:.6f}")
