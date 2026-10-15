<!-- ---
title: "Tracing & Debugging"
description: "Trace every LLM call, tool invocation, and decision point"
icon: "search"
--- -->

# Tracing & Debugging

When an agent does something unexpected, you need to know **exactly why**. Tracing captures the full execution flow — every LLM call, tool invocation, decision point, and intermediate result — so you can reconstruct the agent's reasoning path post-hoc.

This tutorial teaches **observability as a first-class concern** with Python tracing primitives. No external observability service is needed to run the examples.

## Example Intent

These examples follow one debugging loop: **record an agent run → inspect its cost and behavior → locate a failure and replay from a checkpoint**. The trace format is independent of the model provider, so the analysis script can inspect traces produced by either Anthropic or DeepSeek.

- `01_trace_collector.py` records nested agent, LLM, and tool spans, including timing, inputs, outputs, errors, and token counts. With no Anthropic key, it displays a built-in sample trace.
- `02_trace_analysis.py` uses sample traces to calculate metrics, flag repeated or costly behavior, and compare runs. Its cost figures use fixed teaching estimates, not provider-specific prices.
- `03_trace_debugging.py` walks three sample failures (bad search, unsupported claims, and a loop), shows the decision path, suggests fixes, and replays from an error checkpoint. Replay is a dry run when no key is configured for the selected provider.

## 🎯 What You'll Learn

- Build a span-based trace collector with context managers and decorators
- Visualize execution traces as Rich tree hierarchies
- Detect anti-patterns: excessive calls, loops, repeated searches, high token usage
- Compare traces across runs of the same task
- Debug agent failures by walking recorded traces
- Replay agent execution from checkpoints

## 📦 Available Examples

| Script | File | Description |
| ------ | ---- | ----------- |
| Trace Collector | [01_trace_collector.py](01_trace_collector.py) | Build a `TraceCollector` with spans, context managers, decorators |
| Trace Analysis | [02_trace_analysis.py](02_trace_analysis.py) | Load traces, detect anti-patterns, compute metrics |
| Trace Debugging | [03_trace_debugging.py](03_trace_debugging.py) | Failure point identification, decision paths, replay |

## 🚀 Quick Start

> **Prerequisites:** Python 3.11+, API keys, and uv. See [SETUP.md](../../SETUP.md) for full setup instructions.

```bash
uv run --directory 04-testing-evaluation/03-tracing-debugging python 01_trace_collector.py

# Example
uv run --directory 04-testing-evaluation/03-tracing-debugging python 02_trace_analysis.py
```

All scripts include **sample trace data** and work without API keys. The collector and debugger use live Anthropic calls automatically when `ANTHROPIC_API_KEY` is set. The analysis script always works offline.

### Run with DeepSeek

Set `DEEPSEEK_API_KEY` in your environment or the repository root `.env`, then run from the repository root:

```bash
uv run --directory 04-testing-evaluation/03-tracing-debugging python 01_trace_collector.py --provider deepseek
uv run --directory 04-testing-evaluation/03-tracing-debugging python 02_trace_analysis.py
uv run --directory 04-testing-evaluation/03-tracing-debugging python 03_trace_debugging.py --provider deepseek
```

The collector and replay use DeepSeek's [Anthropic-compatible Messages API](https://api-docs.deepseek.com/guides/anthropic_api/) and default to `deepseek-flash`. Set `DEEPSEEK_TRACING_MODEL` to choose another supported model or `DEEPSEEK_ANTHROPIC_BASE_URL` for another compatible endpoint. The DeepSeek flag requires an API key; it does not silently use sample or dry-run mode. Script 02 has no model calls and needs no provider flag. Run the scripts from the repository root so `trace_output.json` is written there.

Or use the [Code Runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) VS Code extension to run the currently open script with a single click.

## 🔑 Key Concepts

### 1. Span-Based Tracing

Every operation is a **span** with timing, inputs, outputs, and child spans:

```python
@dataclass
class Span:
    name: str           # "llm_call_1", "search_knowledge_base"
    span_type: str      # "llm_call", "tool_call", "agent_step"
    start_time: float
    end_time: float
    tokens: dict        # {"input": 150, "output": 80}
    children: list      # Nested child spans
    error: str | None   # Error message if span failed
```

### 2. Context Manager Tracing

The `TraceCollector` uses context managers for automatic span lifecycle:

```python
tracer = TraceCollector()

with tracer.span("answer_question", "agent_step") as root:
    with tracer.span("llm_call", "llm_call") as llm_span:
        response = client.messages.create(...)
        llm_span.tokens = {"input": 150, "output": 80}

    with tracer.span("search", "tool_call") as tool_span:
        results = search_knowledge_base(query)
```

### 3. Anti-Pattern Detection

Automated analysis catches common agent problems:

| Anti-Pattern | Symptom | Typical Cause |
|-------------|---------|---------------|
| Excessive LLM calls | >5 calls for a simple question | Missing stop conditions |
| Repeated searches | Same query searched twice | No result caching |
| High token usage | >2000 tokens for simple task | Verbose prompts or loops |
| Failed tool calls | Tool errors not retried | Missing error handling |
| Very long spans | >10s for a single operation | API timeouts or loops |

### 4. Trace-Based Debugging

When an eval fails, the trace shows the failure path:

```
1. Walk backward from the failed outcome
2. Find the first span with an error or unexpected output
3. Examine the inputs that led to the bad decision
4. Use TraceReplay to re-execute from that checkpoint
```

## ⚠️ Important Considerations

- **Pure Python, not production** — this tutorial teaches concepts. For production, use [Langfuse](https://langfuse.com/), [Datadog](https://www.datadoghq.com/), or [OpenTelemetry](https://opentelemetry.io/)
- **Trace storage grows fast** — in production, sample traces and set retention policies
- **Cost attribution matters** — knowing which step costs the most guides optimization

## 🔗 Resources

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — Industry-standard specification for spans, traces, and context propagation used in production observability
- [Langfuse — Open Source LLM Observability](https://langfuse.com/) — Production tracing for LLM applications with cost tracking, scoring, and prompt management
- [Dapper, a Large-Scale Distributed Systems Tracing Infrastructure — Sigelman et al., 2010](https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/) — Google's seminal paper on span-based distributed tracing that inspired OpenTelemetry
- [The Three Pillars of Observability — Charity Majors](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/ch04.html) — Metrics, logs, and traces as complementary observability signals

## 👉 Next Steps

Once you've mastered tracing, continue to:
- **[Red Teaming & Safety](../04-red-teaming-safety/)** — Test your agents against adversarial attacks
- **Experiment** — Instrument your own agents with the `TraceCollector`
- **Explore** — Connect traces to eval failures from [Tutorial 02](../02-evals/)
