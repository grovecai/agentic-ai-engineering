<!-- ---
title: "AWS Strands"
description: "AWS's agent SDK with native service integration"
icon: "cloud"
--- -->

# AWS Strands

AWS's agent SDK. Model-driven development with native AWS service integration and deployment tooling.

## 🎯 What You'll Learn

- Build agents using AWS Strands architecture
- Integrate with AWS services natively
- Use model-driven development patterns
- Deploy agents with AWS infrastructure tooling

## DeepSeek 示例

Strands Agent、工具与 OpenAI 兼容模型。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/05-aws-strands python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

OpenAIModel 使用 DeepSeek URL；关闭 thinking 以便框架保留有效的多轮工具历史。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://strandsagents.com/docs/user-guide/sdk/model-providers/openai/)。
