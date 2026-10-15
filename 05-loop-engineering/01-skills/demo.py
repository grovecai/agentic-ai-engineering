"""Load a skill only when the task needs it, then run the DeepSeek agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import LOOKUP, Agent, Tool, client_from_env

ROOT = Path(__file__).parent / "skills"


def load_skill(args: dict) -> str:
    name = args.get("name")
    if not isinstance(name, str) or not name.replace("-", "").isalnum():
        raise ValueError("Invalid skill name")
    path = (ROOT / name / "SKILL.md").resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Unknown skill")
    return path.read_text()


SKILL = Tool(
    "load_skill",
    "Load full instructions for a listed skill when relevant.",
    {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["incident-summary"]}},
        "required": ["name"],
    },
    load_skill,
)


if __name__ == "__main__":
    catalog = "Available skill: incident-summary — summarize incidents with timeline and actions."
    print(
        Agent(client_from_env(), [SKILL, LOOKUP]).run(
            "Summarize our incident process.",
            system=f"{catalog} Load the skill when applicable; use local documents and cite IDs.",
        )
    )
