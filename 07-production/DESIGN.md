# DeepSeek production example design

These lessons share a narrow task: answer a question about a local support
policy. The examples demonstrate production controls around that agent.

| Lesson | Executable demonstration |
| --- | --- |
| 12 factors | Config from environment, stateless request handling, explicit tool registry |
| Deployment | HTTP health and chat endpoints, request IDs, container packaging |
| Observability | Structured logs, latency and token counters, trace IDs |
| Cost | Per-request token ceiling, usage accounting, simple response cache |
| Security | Input bounds, tool allowlist, retrieved text treated as data |
| Resilience | Backoff for transient API errors and bounded attempts |

`service.py` holds the shared application boundary and accepts a fake model
client for offline verification. Live calls use `DEEPSEEK_API_KEY`,
`DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL`. The deployment example is a local
reference deployment; production operators still need TLS, authentication,
secrets management, and network policy at their platform boundary.
