<!-- ---
title: "Hooks & Lifecycle"
description: "Intercept the agent loop at pre/post tool-use and stop events with pluggable callbacks"
icon: "git-branch"
--- -->

# Hooks & Lifecycle

Intercept the agent loop at its seams. A dependency-free hook registry lets you add logging, guardrails, and approval gates as pluggable callbacks instead of rewriting the loop — the extensibility surface every later tutorial plugs into.

## 🎯 What You'll Learn

- Design a hook registry mapping lifecycle events to callables
- Handle `pre_tool_use`, `post_tool_use`, and `stop` events
- Return allow / deny / replace decisions from a pre-tool-use hook
- Re-express blocked-command guardrails as a built-in hook

## DeepSeek 示例

工具调用前后与结束时触发 hooks；拒绝非允许工具。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 02-hooks-lifecycle/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

before hook 可以拒绝工具执行；after hook 只在工具成功后运行；stop hook 在退出时运行。
