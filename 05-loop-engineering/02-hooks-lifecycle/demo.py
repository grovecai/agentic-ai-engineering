"""Show pre-tool, post-tool, and stop hooks around the same agent loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import LOOKUP, Agent, Hooks, client_from_env


def before(name: str, args: dict) -> None:
    if name != "lookup":
        raise PermissionError(f"Tool {name} is not approved")
    print(f"before {name}: {args}")


def after(name: str, args: dict, result: object) -> None:
    print(f"after {name}: {str(result)[:80]}")


if __name__ == "__main__":
    hooks = Hooks(before=[before], after=[after], stop=[lambda task: print(f"stop: {task}")])
    print(Agent(client_from_env(), [LOOKUP], hooks).run("What is the deployment rollback rule?"))
