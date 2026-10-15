"""Minimal HTTP deployment surface for the support agent."""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import Settings, SupportService, authorize

app = FastAPI(title="DeepSeek support demo")
_service: SupportService | None = None


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/answer")
def answer(request: Question, x_demo_key: str | None = Header(default=None)) -> dict:
    global _service
    try:
        authorize(x_demo_key, os.getenv("DEMO_SERVER_API_KEY"))
        if _service is None:
            _service = SupportService(Settings.from_env())
        return _service.answer(request.text)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
