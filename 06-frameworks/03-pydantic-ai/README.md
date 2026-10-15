<!-- ---
title: "Pydantic AI"
description: "Type-safe agent framework built on Pydantic"
icon: "shield"
--- -->

# Pydantic AI

Type-safe agent framework built on Pydantic. Strong typing, dependency injection, and structured outputs as first-class citizens.

## 🎯 What You'll Learn

- Define agents with Pydantic models for type safety
- Use dependency injection for testable agent code
- Get structured outputs validated at runtime
- Leverage Python's type system for agent development

## DeepSeek 示例

Pydantic AI 的类型化工具参数与 Agent。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/03-pydantic-ai python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

以 Literal 限制工具 ID，使用 OpenAIChatModel 的兼容端点。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://pydantic.dev/docs/ai/models/compatible-apis/)。
