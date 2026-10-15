"""Bound transient retries, then optionally fall back to a second model."""

import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import TRANSIENT, Settings, SupportService

if __name__ == "__main__":
    service = SupportService(Settings.from_env())
    try:
        print(service.answer("What is the shipping policy?"))
    except TRANSIENT as exc:
        print(
            f"Request failed after {service.settings.max_attempts} attempts: {type(exc).__name__}"
        )
        fallback_model = os.getenv("DEEPSEEK_FALLBACK_MODEL")
        if fallback_model:
            fallback = SupportService(replace(service.settings, model=fallback_model))
            print("Fallback answer:", fallback.answer("What is the shipping policy?"))
