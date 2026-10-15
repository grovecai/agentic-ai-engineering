"""Delegate bounded, independent research tasks to isolated child agents."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import LOOKUP, Agent, Tool, client_from_env

MAX_CHILDREN = 2


def research(args: dict) -> str:
    questions = args.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_CHILDREN:
        raise ValueError("Provide one or two research questions")
    if any(not isinstance(q, str) or len(q) > 300 for q in questions):
        raise ValueError("Each question must be short text")

    def child(question: str) -> tuple[str, dict]:
        agent = Agent(client_from_env(), [LOOKUP], max_steps=3)
        answer = agent.run(
            question,
            "Use lookup to research this subtask. Return a brief factual summary with document ID.",
        )
        return answer, agent.last_usage

    with ThreadPoolExecutor(max_workers=MAX_CHILDREN) as pool:
        results = list(pool.map(child, questions))
    return "\n".join(
        f"Subtask {i + 1}: {answer} (tokens: {usage['input']} in, {usage['output']} out)"
        for i, (answer, usage) in enumerate(results)
    )


DELEGATE = Tool(
    "delegate_research",
    "Ask up to two isolated child agents to research independent questions.",
    {
        "type": "object",
        "properties": {"questions": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
        "required": ["questions"],
    },
    research,
)

if __name__ == "__main__":
    parent = Agent(client_from_env(), [DELEGATE], max_steps=3)
    print(
        parent.run(
            "Compare the deployment and incident runbooks. Delegate one question about each, then summarize."
        )
    )
    print("Parent usage:", parent.last_usage)
