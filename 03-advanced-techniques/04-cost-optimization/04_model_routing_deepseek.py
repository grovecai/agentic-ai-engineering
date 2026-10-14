"""Route easy tasks to DeepSeek Flash and hard tasks to Pro."""

import argparse

from rich.console import Console
from rich.markdown import Markdown

from deepseek_cost_common import FLASH, PRICE_TIER, PRO, Usage, client, cost

SAMPLE_TASKS = [
    "What is the capital of France?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Design a microservices architecture for a multiplayer game with 100,000 concurrent users.",
    "Compare event sourcing with CRUD for an audited financial transaction system.",
]


class ModelRouter:
    def __init__(self) -> None:
        self.api = client()
        self.routed_total = 0.0
        self.baseline_total = 0.0
        self.count = 0

    def request(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, Usage]:
        response = self.api.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = response.choices[0].message.content
        if not answer:
            raise ValueError(f"{model} returned an empty answer")
        return answer, Usage.from_response(response)

    def run(self, task: str) -> tuple[str, str, str, float, float, float]:
        classification, classifier_usage = self.request(
            FLASH,
            [
                {"role": "system", "content": (
                    "Classify the task. Reply with one word: easy or hard. "
                    "Easy: factual lookup, simple conversion, basic math. "
                    "Hard: design, analysis, multi-step reasoning, comparison."
                )},
                {"role": "user", "content": task},
            ],
            8,
        )
        difficulty = classification.strip().lower()
        if difficulty not in {"easy", "hard"}:
            difficulty = "hard"  # Favor quality when classification is unclear.
        model = FLASH if difficulty == "easy" else PRO
        answer, answer_usage = self.request(model, [{"role": "user", "content": task}], 700)

        classifier_cost = cost(FLASH, classifier_usage)
        routed = classifier_cost + cost(model, answer_usage)
        # Counterfactual estimate: same answer token counts, priced as Pro. An
        # actual Pro response may have a different length or cache hit rate.
        baseline = cost(PRO, answer_usage)
        self.routed_total += routed
        self.baseline_total += baseline
        self.count += 1
        return difficulty, model, answer, classifier_cost, routed, baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run four sample tasks")
    args = parser.parse_args()
    console = Console()
    router = ModelRouter()
    console.print(
        f"DeepSeek routing: Flash classifier -> Flash (easy) / Pro (hard). "
        f"Estimated at {PRICE_TIER} prices."
    )
    tasks = iter(SAMPLE_TASKS) if args.demo else None
    while True:
        if tasks is None:
            task = console.input("\n[green]Task (quit to exit):[/green] ").strip()
            if task.lower() in {"quit", "exit", ""}:
                break
        else:
            task = next(tasks, None)
            if task is None:
                break
            console.print(f"\n[green]Task:[/green] {task}")
        difficulty, model, answer, classifier_cost, routed, baseline = router.run(task)
        console.print(f"{difficulty} -> {model}")
        console.print(Markdown(answer))
        console.print(
            f"Estimated classifier: ${classifier_cost:.6f}; routed: ${routed:.6f}; "
            f"all-Pro baseline: ${baseline:.6f}"
        )
    if router.count:
        console.print(
            f"\n{router.count} tasks. Estimated routed: ${router.routed_total:.6f}; "
            f"all-Pro baseline: ${router.baseline_total:.6f}; "
            f"difference: ${router.baseline_total - router.routed_total:.6f}"
        )


if __name__ == "__main__":
    main()
