<!-- ---
title: "Semantic Kernel"
description: "Microsoft's AI orchestration SDK with plugin architecture"
icon: "puzzle"
--- -->

# Semantic Kernel

Microsoft's AI orchestration SDK. Plugin architecture, planner system, and deep integration with Azure services.

## 🎯 What You'll Learn

- Build agents using Semantic Kernel's plugin system
- Use the planner for automatic task decomposition
- Integrate with Azure AI services seamlessly
- Create skills and functions for agent capabilities

## DeepSeek 示例

Semantic Kernel ChatCompletionAgent 与插件。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/09-semantic-kernel python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

ChatCompletionAgent 通过 OpenAIChatCompletion 服务调用本地插件。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/chat-completion-agent?pivots=programming-language-python)。
