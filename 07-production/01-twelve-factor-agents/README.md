<!-- ---
title: "12-Factor Agents"
description: "Principles for building production-grade agents"
icon: "list"
--- -->

# 12-Factor Agents

Principles for building production-grade agents. Inspired by the 12-factor app methodology, adapted for the unique challenges of LLM-powered systems.

## 🎯 What You'll Learn

- Apply 12-factor principles to agent architecture
- Design agents for portability and scalability
- Configure agents through environment, not code
- Build production-ready agents from day one

## DeepSeek 示例

从环境加载配置，使用无状态的请求处理与明确的工具列表。

从仓库根目录运行：

```bash
uv run --directory 07-production python 01-twelve-factor-agents/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

本例展示环境配置、独立请求状态和显式工具清单。12-factor 是设计检查清单，不代表单个脚本覆盖全部运行环境要求。
