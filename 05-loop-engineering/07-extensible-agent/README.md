<!-- ---
title: "Extensible Agent"
description: "Capstone — one agent wired with hooks, sandbox, MCP, subagents, and compaction"
icon: "boxes"
--- -->

# Extensible Agent 🏆

**Capstone.** Compose every primitive from this module into a single extensible agent: hooks for control, a sandboxed executor for safety, MCP-discovered tools for reach, subagent delegation for scale, and automatic compaction for endurance.

## 🎯 What You'll Learn

- Wire hooks, sandboxing, MCP, subagents, and compaction into one coherent loop
- Keep the composition clean — each primitive stays an independent, swappable layer
- Drive an end-to-end task that exercises every control surface at once
- Recognize which harness concerns a framework would (and wouldn't) abstract away

## DeepSeek 示例

组合技能、工具与 hooks，并展示统一的步数和用量限制。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 07-extensible-agent/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

capstone 复用 01 的 `SkillRegistry`：启动时扫描技能目录，将简短目录加入 system prompt，并动态注册 `load_skill` / `read_skill_file`。同一 agent 还注册 MCP、子 agent 和固定沙箱工具，通过 hook 控制调用并在第二轮后压缩历史。运行输出会列出实际加载的技能。
