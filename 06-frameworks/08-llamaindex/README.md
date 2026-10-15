<!-- ---
title: "LlamaIndex"
description: "Data-centric agent framework with strong RAG primitives"
icon: "database"
--- -->

# LlamaIndex

Data-centric agent framework. Strong RAG primitives, knowledge graph integration, and document-aware agents.

## 🎯 What You'll Learn

- Build data-aware agents with LlamaIndex
- Use built-in RAG and retrieval pipelines
- Integrate knowledge graphs for complex reasoning
- Create agents that work naturally with documents

## DeepSeek 示例

LlamaIndex FunctionAgent 与 FunctionTool。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/08-llamaindex python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

OpenAILike 提供兼容客户端，FunctionAgent 管理工具调用。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://docs.llamaindex.ai/en/stable/)。
