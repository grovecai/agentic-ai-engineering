"""Compact complete conversation turns while preserving tool-call/result pairs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import MODEL, client_from_env


def complete_turns(messages: list[dict]) -> list[list[dict]]:
    """Group user turns; reject an orphaned tool response or unresolved call."""
    turns: list[list[dict]] = []
    current: list[dict] = []
    pending: set[str] = set()
    for message in messages:
        role = message["role"]
        if role == "user":
            if pending:
                raise ValueError("Cannot split a tool call from its result")
            if current:
                turns.append(current)
            current = [message]
        elif role == "assistant":
            current.append(message)
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Orphaned tool result")
            pending.remove(message["tool_call_id"])
            current.append(message)
        else:
            raise ValueError(f"Unexpected role: {role}")
    if pending:
        raise ValueError("Unresolved tool call")
    if current:
        turns.append(current)
    return turns


def compact(messages: list[dict], client: object, keep_last: int = 1) -> list[dict]:
    system, *history = messages
    if system["role"] != "system":
        raise ValueError("First message must be system")
    turns = complete_turns(history)
    if len(turns) <= keep_last:
        return messages
    old = [message for turn in turns[:-keep_last] for message in turn]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Summarize prior facts, decisions, open questions and cited document IDs. Do not invent facts.",
            },
            {"role": "user", "content": json.dumps(old, ensure_ascii=False)},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    summary = response.choices[0].message.content
    if not summary:
        raise RuntimeError("Compaction summary was empty")
    retained = [message for turn in turns[-keep_last:] for message in turn]
    return [
        system,
        {"role": "user", "content": f"Earlier conversation summary:\n{summary}"},
        *retained,
    ]


if __name__ == "__main__":
    history = [
        {"role": "system", "content": "Answer using local operations facts."},
        {"role": "user", "content": "What is the runbook?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": '{"doc_id":"runbook"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "runbook: check health and error rate",
        },
        {
            "role": "assistant",
            "content": "The runbook says to check health and error rate [runbook].",
        },
        {"role": "user", "content": "What about incidents?"},
        {"role": "assistant", "content": "I will check the incident document."},
    ]
    reduced = compact(history, client_from_env())
    print(json.dumps(reduced, indent=2, ensure_ascii=False))
