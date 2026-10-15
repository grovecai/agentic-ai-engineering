"""The identical task and lookup function used by all framework examples."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
QUESTION = "When should we roll back a deployment, and what should an incident owner record?"
INSTRUCTIONS = "Use lookup_document for local facts. Cite document IDs runbook and incident."
DOCUMENTS = {
    "runbook": "After deployment, check health, latency, and error rate. Roll back if thresholds are breached.",
    "incident": "Record a trace ID, impact, owner, mitigation, and follow-up action.",
}


def key() -> str:
    value = os.getenv("DEEPSEEK_API_KEY")
    if not value:
        raise SystemExit("Set DEEPSEEK_API_KEY to run this example")
    return value


def lookup_document(doc_id: str) -> str:
    """Read a local operations document by its ID: runbook or incident."""
    return f"{doc_id}: {DOCUMENTS.get(doc_id, 'Unknown document ID')}"
