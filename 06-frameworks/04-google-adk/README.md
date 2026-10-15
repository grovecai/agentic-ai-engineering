<!-- ---
title: "Google ADK"
description: "Google's Agent Development Kit with multi-agent orchestration"
icon: "cpu"
--- -->

# Google ADK

Google's Agent Development Kit. Multi-agent orchestration with built-in tool support and Gemini integration.

## 🎯 What You'll Learn

- Build agents using Google's ADK architecture
- Leverage native Gemini model integration
- Use built-in tool definitions and execution
- Orchestrate multi-agent workflows with ADK primitives

## DeepSeek 示例

ADK Runner、会话及 LiteLLM 的 DeepSeek 桥接。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/04-google-adk python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

通过 ADK 的 LiteLLM 模型桥接；会话仅存于内存。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://adk.dev/agents/models/litellm/)。
