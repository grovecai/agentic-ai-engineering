<!-- ---
title: "No Framework"
description: "The raw SDK baseline — pure client code with no abstractions"
icon: "code"
--- -->

# No Framework

The raw SDK baseline. Pure Anthropic/OpenAI client code with no abstractions. This is your reference implementation — everything else is measured against it.

## 🎯 What You'll Learn

- Build a complete agent using only provider SDKs
- Understand exactly what frameworks abstract away
- Establish a baseline for comparing framework trade-offs
- Make informed decisions about when abstraction helps

## DeepSeek 示例

OpenAI 兼容 SDK 的有界工具循环，作为比较基线。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/01-no-framework python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

直接使用 DeepSeek OpenAI 兼容端点；工具参数会在本地解析并验证。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://api-docs.deepseek.com/guides/tool_calls/)。
