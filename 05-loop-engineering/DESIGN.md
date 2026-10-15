# DeepSeek example design

The seven lessons build one small research agent in stages. Each lesson has a
standalone `demo.py` entry point, and shared primitives live in `harness.py`.
The agent can answer questions from a fixed local document store; it never
needs a web account or a writable external service.

| Lesson | Concrete behavior | Main boundary |
| --- | --- | --- |
| Skills | Discover a catalog, load one instruction file on demand | Resolve paths inside the skills directory |
| Hooks | Observe before/after calls, reject a forbidden call | Hook rejection stops tool execution |
| Sandboxing | Execute an allowlisted Python snippet in a subprocess | Resource limits are a teaching baseline, not strong isolation |
| MCP | Discover and call a local stdio MCP server | Tool schemas and results cross a process boundary |
| Subagents | Delegate two independent research questions | Bound depth, fan-out, and call count |
| Compaction | Summarize completed turns and retain the current turn | Keep each tool call with its result |
| Capstone | Compose the above in one bounded agent | Shared state and limits remain explicit |

All live examples use `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default
`https://api.deepseek.com`) and `DEEPSEEK_MODEL` (default `deepseek-flash`).
The demos use the OpenAI-compatible Chat Completions API with thinking disabled
for predictable tool-call history. Offline checks exercise the boundaries
without making billable calls.
