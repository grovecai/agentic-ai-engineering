"""Emit structured per-request logs and aggregate model token counts."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Settings, SupportService

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    service = SupportService(Settings.from_env())
    print(json.dumps(service.answer("How long does shipping take?"), ensure_ascii=False))
    print(json.dumps(service.metrics))
