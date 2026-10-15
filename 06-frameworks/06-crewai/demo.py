"""CrewAI role and task using a DeepSeek-compatible LLM and custom tool."""

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "CREWAI_TESTING", "true"
)  # Suppress first-run trace collection prompt in this demo.

from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


@tool("lookup_document")
def lookup(doc_id: str) -> str:
    """Read the local runbook or incident document by ID."""
    return lookup_document(doc_id)


if __name__ == "__main__":
    llm = LLM(model=f"openai/{MODEL}", base_url=BASE_URL, api_key=key())
    researcher = Agent(
        role="Operations researcher",
        goal="Answer from local documents",
        backstory="You cite document IDs for every local fact.",
        llm=llm,
        tools=[lookup],
        verbose=False,
    )
    task = Task(
        description=f"{INSTRUCTIONS}\nQuestion: {QUESTION}",
        expected_output="A concise answer citing runbook and incident",
        agent=researcher,
    )
    print(Crew(agents=[researcher], tasks=[task], verbose=False).kickoff())
