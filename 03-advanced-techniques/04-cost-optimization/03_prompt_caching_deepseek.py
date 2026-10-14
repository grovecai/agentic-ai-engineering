"""DeepSeek's automatic prefix cache in a multi-turn support conversation."""

from rich.console import Console
from rich.markdown import Markdown

from deepseek_cost_common import FLASH, PRICE_TIER, Usage, client, cost

POLICY = """You are a support agent for TechFlow Solutions. Answer only from this policy;
say when it does not cover a question.

Annual subscriptions can be refunded within 30 days of purchase. After 30 days,
the remaining balance becomes account credit valid for 12 months. Monthly plans
can be cancelled at any time but unused days are not refunded. Enterprise contracts
follow their individual agreements. Hardware can be returned within 30 days of
delivery in original packaging. Defective hardware is replaced under warranty.
Standard shipping takes 5-7 business days and is free above $50; otherwise it
costs $7.99. Express shipping takes 2-3 business days and costs $14.99.
Verified educational institutions and registered nonprofits receive a 40% discount.
Basic support responds by email in 24-48 hours. Pro includes priority email and
live chat. Enterprise includes a dedicated account manager and phone support.
"""


def main() -> None:
    console = Console()
    api = client()
    history: list[dict[str, str]] = [{"role": "system", "content": POLICY}]
    total_cost = 0.0
    baseline_cost = 0.0
    console.print(
        f"DeepSeek automatic context cache ({FLASH}, {PRICE_TIER} prices). "
        "Ask several questions in this session; type 'quit' to exit.\n"
        "The repeated policy and prior conversation can be cached; each answer is generated anew."
    )
    while True:
        question = console.input("\n[green]You:[/green] ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        history.append({"role": "user", "content": question})
        try:
            response = api.chat.completions.create(
                model=FLASH,
                messages=history,
                max_tokens=500,
                extra_body={"thinking": {"type": "disabled"}},
            )
            answer = response.choices[0].message.content
            if not answer:
                raise ValueError("DeepSeek returned an empty answer")
            usage = Usage.from_response(response)
        except Exception:
            history.pop()
            raise
        history.append({"role": "assistant", "content": answer})
        actual = cost(FLASH, usage)
        baseline = cost(FLASH, usage, ignore_cache=True)
        total_cost += actual
        baseline_cost += baseline
        console.print(Markdown(answer))
        console.print(
            f"Input: {usage.input_tokens:,} tokens "
            f"({usage.cached_tokens:,} cached, {usage.uncached_tokens:,} uncached); "
            f"output: {usage.output_tokens:,}. Estimated call: ${actual:.6f} "
            f"vs ${baseline:.6f} without cache."
        )
        console.print(
            f"Estimated total: ${total_cost:.6f}; no-cache baseline: "
            f"${baseline_cost:.6f}; savings: ${baseline_cost - total_cost:.6f}"
        )


if __name__ == "__main__":
    main()
