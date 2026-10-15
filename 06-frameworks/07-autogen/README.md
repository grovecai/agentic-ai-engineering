<!-- ---
title: "AutoGen"
description: "Microsoft's multi-agent conversation framework"
icon: "message-circle"
--- -->

# AutoGen

Microsoft's multi-agent conversation framework. Agents communicate through structured conversations with configurable interaction patterns.

## 🎯 What You'll Learn

- Set up multi-agent conversations with AutoGen
- Configure interaction patterns and termination conditions
- Use human-in-the-loop conversation flows
- Build autonomous agent teams for complex tasks

## DeepSeek 示例

AutoGen AssistantAgent 和模型客户端。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/07-autogen python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

OpenAIChatCompletionClient 需要为非 OpenAI 模型声明 model_info。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-clients.html)。
