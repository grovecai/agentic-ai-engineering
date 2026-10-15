<!-- ---
title: "LangGraph"
description: "Graph-based agent orchestration from LangChain"
icon: "git-branch"
--- -->

# LangGraph

Graph-based agent orchestration from LangChain. Define agents as nodes and edges with built-in state management and persistence.

## 🎯 What You'll Learn

- Model agent workflows as directed graphs
- Use built-in state management and checkpointing
- Implement conditional edges for dynamic routing
- Compare graph-based approach to linear orchestration

## DeepSeek 示例

LangGraph ReAct graph 与本地 lookup 工具。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/02-langgraph python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

显式定义模型节点、工具节点、条件边和内存 checkpoint。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://docs.langchain.com/oss/python/langgraph/overview)。
