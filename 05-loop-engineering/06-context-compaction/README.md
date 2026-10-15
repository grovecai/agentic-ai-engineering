<!-- ---
title: "Context Compaction"
description: "Keep long-running loops alive by compacting their own history mid-run"
icon: "scissors"
--- -->

# Context Compaction

Long-running loops eventually outgrow the context window. Compaction lets the agent summarize its own old turns mid-run — pinning the system prompt and task, and never splitting a tool call from its result — so the loop survives many iterations.

## 🎯 What You'll Learn

- Trigger compaction as a loop-lifecycle event with a `should_compact()` check
- Summarize old turns while pinning the system prompt and current task
- Preserve `tool_use`/`tool_result` pairing to avoid API errors
- Handle compaction on the OpenAI Responses API by resending a compacted input

## DeepSeek 示例

按完整回合压缩旧历史，保证工具调用与结果不被拆开。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 06-context-compaction/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

旧回合经 DeepSeek 总结，最新回合完整保留；未配对的工具调用会被拒绝。
