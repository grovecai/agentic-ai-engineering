<!-- ---
title: "Sandboxing"
description: "Run agent-generated code and tool calls in isolation with resource limits and timeouts"
icon: "shield"
--- -->

# Sandboxing

Run agent-generated code and tool calls without handing over the machine. Start with a portable subprocess sandbox — stripped environment, resource limits, ephemeral working directory, and timeouts — then reach for real container isolation when you need a hard boundary.

## 🎯 What You'll Learn

- Contain execution with `subprocess`, `resource.setrlimit`, and a stripped environment
- Enforce CPU, memory, and wall-clock limits on agent-run code
- Add an optional Docker (`--network none`) tier for a real isolation boundary
- Understand what each tier actually protects against — and what it doesn't

## DeepSeek 示例

用隔离的子进程运行固定演示代码，限制时间、内存与输出。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 03-sandboxing/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
The subprocess limits here are a teaching baseline. Use a container or dedicated sandbox service for hostile code.

### 实现说明

本地子进程只演示资源限制。它不构成对恶意代码的强隔离；生产需要专门的容器或沙箱。
