"""Agent Skills: catalog -> load SKILL.md -> read supporting file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import LOOKUP, Agent, Hooks, client_from_env
from skills import SkillRegistry

SKILLS_ROOT = Path(__file__).parent / "skills"


def build_agent(client: object) -> tuple[Agent, SkillRegistry]:
    registry = SkillRegistry(SKILLS_ROOT)
    hooks = Hooks(before=[lambda name, args: print(f"tool: {name} {args}")])
    return Agent(client, [*registry.tools(), LOOKUP], hooks=hooks), registry


if __name__ == "__main__":
    agent, registry = build_agent(client_from_env())
    print("Catalog visible before the request:\n" + registry.catalog_prompt())
    answer = agent.run(
        "Summarize our incident process using the relevant skill and its template.",
        system=registry.catalog_prompt() + "\nUse local documents for factual claims and cite IDs.",
    )
    print(answer)
    print("Skills loaded on demand:", sorted(registry.loaded))
