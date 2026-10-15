"""OpenAI-compatible SDK baseline: one bounded tool loop."""

import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_document",
        "description": "Read runbook or incident by ID.",
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "enum": ["runbook", "incident"]}},
            "required": ["doc_id"],
        },
    },
}


def main() -> None:
    client = OpenAI(api_key=key(), base_url=BASE_URL)
    messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": QUESTION}]
    for _ in range(4):
        reply = (
            client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[TOOL],
                extra_body={"thinking": {"type": "disabled"}},
            )
            .choices[0]
            .message
        )
        if not reply.tool_calls:
            print(reply.content)
            return
        messages.append(reply.model_dump(exclude_none=True))
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = (
                lookup_document(args.get("doc_id", ""))
                if call.function.name == "lookup_document"
                else "Unknown tool"
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    raise RuntimeError("Exceeded four model steps")


if __name__ == "__main__":
    main()
