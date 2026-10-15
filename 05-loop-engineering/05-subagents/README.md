<!-- ---
title: "Subagents & Delegation"
description: "Spawn child loops with isolated context that run a subtask and return a summary"
icon: "users"
--- -->

# Subagents & Delegation

Let a parent loop delegate. A `spawn_subagent` tool builds a fresh agent with its own message history, runs a subtask to completion, and returns only a summary — keeping the parent's context clean while parallelizing work.

## 🎯 What You'll Learn

- Model a parent agent spawning child agents with isolated context
- Return compact summaries instead of full child transcripts
- Enforce delegation depth and fan-out limits
- Aggregate child token usage into the parent's report

## DeepSeek 示例

父 agent 将两个独立问题交给有独立上下文的子 agent，并汇总 token 用量。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 05-subagents/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

子 agent 各自有独立对话，最多并行两个，每个最多三个模型步骤。
